//! poker_native — cœur MCCFR compilé pour PokerIA.
//! Expose : NativeStore (table + itérations), PyHand (moteur, pour les tests
//! de parité), et les briques (évaluateur, équité, buckets).

mod bucket;
mod equity;
mod eval;
mod game;
mod rng;
mod traverse;

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::rng::Rng;
use crate::traverse::Nodes;

/// Table de nœuds + compteur d'itérations, vivant côté Rust.
#[pyclass]
struct NativeStore {
    nodes: Nodes,
    #[pyo3(get, set)]
    iterations: u64,
    #[pyo3(get)]
    version: u8,
}

#[pymethods]
impl NativeStore {
    #[new]
    fn new(version: u8) -> Self {
        NativeStore { nodes: Nodes::default(), iterations: 0, version }
    }

    /// Charge la table depuis (clés, valeurs (N,10) float32).
    fn load_nodes(&mut self, keys: Vec<String>, values: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = values.as_array();
        if arr.shape() != [keys.len(), 10] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "values doit être de forme (len(keys), 10)"));
        }
        self.nodes.reserve(keys.len());
        for (i, key) in keys.into_iter().enumerate() {
            let mut node = [0f32; 10];
            for j in 0..10 {
                node[j] = arr[[i, j]];
            }
            self.nodes.insert(key, node);
        }
        Ok(())
    }

    /// Exporte la table : (clés, valeurs (N,10) float32).
    fn export_nodes<'py>(&self, py: Python<'py>) -> PyResult<(Vec<String>, Bound<'py, PyArray2<f32>>)> {
        let n = self.nodes.len();
        let mut keys = Vec::with_capacity(n);
        let mut flat = Vec::with_capacity(n * 10);
        for (k, v) in &self.nodes {
            keys.push(k.clone());
            flat.extend_from_slice(v);
        }
        let arr = numpy::PyArray1::from_vec(py, flat).reshape([n, 10])?;
        Ok((keys, arr))
    }

    /// Fait tourner `n` itérations MCCFR (le GIL est relâché pendant le calcul).
    fn run(&mut self, py: Python<'_>, n: u64, seed: u64, n_sims: usize) {
        let nodes = &mut self.nodes;
        let version = self.version;
        let start = self.iterations;
        py.allow_threads(move || {
            let mut rng = Rng::new(seed);
            for t in start..start + n {
                traverse::run_iteration(nodes, t, &mut rng, n_sims, version);
            }
        });
        self.iterations += n;
    }

    /// Un cycle parallèle EN MÉMOIRE : `workers` threads calculent chacun `chunk`
    /// itérations sur une copie de la table courante, puis on fusionne — sans
    /// jamais écrire de fichiers ouvriers sur le disque. Remplace le découpage
    /// en processus + pickles de train_cfr_parallel.py pour le moteur natif.
    fn run_parallel(&mut self, py: Python<'_>, workers: usize, chunk: u64, base_seed: u64, n_sims: usize) {
        let version = self.version;
        let start = self.iterations;
        let base = &self.nodes;
        let merged = py.allow_threads(|| {
            let locals: Vec<traverse::Nodes> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..workers)
                    .map(|w| {
                        s.spawn(move || {
                            let mut local = base.clone();
                            let mut rng = Rng::new(base_seed.wrapping_add(w as u64));
                            for t in start..start + chunk {
                                traverse::run_iteration(&mut local, t, &mut rng, n_sims, version);
                            }
                            local
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            traverse::merge(base, locals)
        });
        self.nodes = merged;
        self.iterations += workers as u64 * chunk;
    }

    fn __len__(&self) -> usize {
        self.nodes.len()
    }
}

/// Moteur de jeu exposé pour les tests de parité (rejeu de séquences).
#[pyclass]
struct PyHand {
    inner: game::Hand,
}

#[pymethods]
impl PyHand {
    #[new]
    fn new(deck: Vec<u8>, button: usize) -> PyResult<Self> {
        if deck.len() < 9 {
            return Err(pyo3::exceptions::PyValueError::new_err("deck : 9 cartes minimum"));
        }
        Ok(PyHand { inner: game::Hand::new(&deck, button) })
    }

    fn step(&mut self, action: u8) {
        self.inner.step(action);
    }

    fn legal_actions(&self) -> Vec<u8> {
        self.inner.legal_actions()
    }

    #[getter] fn stacks(&self) -> (i32, i32) { (self.inner.stacks[0], self.inner.stacks[1]) }
    #[getter] fn invested(&self) -> (i32, i32) { (self.inner.invested[0], self.inner.invested[1]) }
    #[getter] fn bets(&self) -> (i32, i32) { (self.inner.bets[0], self.inner.bets[1]) }
    #[getter] fn pot(&self) -> i32 { self.inner.pot() }
    #[getter] fn street(&self) -> u8 { self.inner.street }
    #[getter] fn to_act(&self) -> usize { self.inner.to_act }
    #[getter] fn terminal(&self) -> bool { self.inner.terminal }
    #[getter] fn showdown(&self) -> bool { self.inner.showdown }
    #[getter] fn winner(&self) -> i8 { self.inner.winner }
    #[getter] fn payoffs(&self) -> (i32, i32) { (self.inner.payoffs[0], self.inner.payoffs[1]) }
    #[getter] fn history(&self) -> Vec<(u8, u8, u8)> { self.inner.history.clone() }
    #[getter] fn history_key(&self) -> String { bucket::history_key(&self.inner) }
}

/// Score d'une main de 7 cartes — parité exacte avec evaluate_hand.
#[pyfunction]
fn eval7(cards: Vec<u8>) -> PyResult<u32> {
    if cards.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err("7 cartes attendues"));
    }
    let mut arr = [0u8; 7];
    arr.copy_from_slice(&cards);
    Ok(eval::evaluate7(&arr))
}

/// Équité Monte-Carlo contre main aléatoire.
#[pyfunction]
#[pyo3(name = "equity")]
fn py_equity(hole: Vec<u8>, board: Vec<u8>, n_sims: usize, seed: u64) -> f64 {
    let h = [hole[0], hole[1]];
    let mut rng = Rng::new(seed);
    equity::equity_vs_random(&h, &board, n_sims, &mut rng)
}

/// Classe préflop canonique ("AA", "AKs", "72o").
#[pyfunction]
#[pyo3(name = "preflop_class")]
fn py_preflop_class(hole: Vec<u8>) -> String {
    bucket::preflop_class(&[hole[0], hole[1]])
}

/// Drapeau de tirage (2 couleur, 1 suite, 0 rien).
#[pyfunction]
#[pyo3(name = "draw_flag")]
fn py_draw_flag(hole: Vec<u8>, board: Vec<u8>) -> u8 {
    bucket::draw_flag(&[hole[0], hole[1]], &board)
}

#[pymodule]
fn poker_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeStore>()?;
    m.add_class::<PyHand>()?;
    m.add_function(wrap_pyfunction!(eval7, m)?)?;
    m.add_function(wrap_pyfunction!(py_equity, m)?)?;
    m.add_function(wrap_pyfunction!(py_preflop_class, m)?)?;
    m.add_function(wrap_pyfunction!(py_draw_flag, m)?)?;
    Ok(())
}
