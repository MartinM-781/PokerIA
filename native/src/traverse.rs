//! Traversée MCCFR external-sampling — port exact de poker_ai/cfr.py.
//! Nœud = [f32; 14] : [0..7] régrets, [7..14] somme de stratégie (abstraction
//! v3 à 7 actions : fold, call, ½, pot, tapis, ¼, ⅓).

use std::collections::HashMap;

use crate::bucket::{card_bucket_v2, infoset_key, preflop_class};
use crate::game::{Hand, BB, PREFLOP};
use crate::rng::Rng;

pub const N_ACT: usize = 7; // actions de l'abstraction v3
pub const NODE: usize = 2 * N_ACT; // régrets [0..7] + stratégie [7..14]

pub type Nodes = HashMap<String, [f32; NODE]>;

/// Cache des buckets par (joueur, street) pour une donne — l'équité coûte cher.
pub struct DealCache {
    entries: [[Option<String>; 4]; 2],
    version: u8,
}

impl DealCache {
    pub fn new(version: u8) -> Self {
        DealCache { entries: Default::default(), version }
    }

    fn bucket(&mut self, hand: &Hand, player: usize, rng: &mut Rng, n_sims: usize) -> String {
        let street = hand.street as usize;
        if let Some(b) = &self.entries[player][street] {
            return b.clone();
        }
        let b = if self.version >= 2 {
            card_bucket_v2(hand, player, rng, n_sims)
        } else if hand.street == PREFLOP {
            preflop_class(&hand.hole[player])
        } else {
            // v1 (équité seule) n'est plus entraînée en natif ; v2 par défaut.
            card_bucket_v2(hand, player, rng, n_sims)
        };
        self.entries[player][street] = Some(b.clone());
        b
    }
}

pub fn traverse(
    hand: &Hand,
    traverser: usize,
    nodes: &mut Nodes,
    rng: &mut Rng,
    cache: &mut DealCache,
    n_sims: usize,
) -> f64 {
    if hand.terminal {
        return hand.payoffs[traverser] as f64 / BB as f64;
    }
    let p = hand.to_act;
    let legal = hand.legal_actions();
    let bucket = cache.bucket(hand, p, rng, n_sims);
    let key = infoset_key(hand, p, &bucket);

    // Stratégie par regret matching (copie locale : les enfants mutent la table)
    let mut sigma = [0f64; N_ACT];
    {
        let node = nodes.entry(key.clone()).or_insert([0f32; NODE]);
        let mut total = 0f64;
        for &a in &legal {
            let v = node[a as usize].max(0.0) as f64;
            sigma[a as usize] = v;
            total += v;
        }
        if total > 1e-12 {
            for &a in &legal {
                sigma[a as usize] /= total;
            }
        } else {
            let u = 1.0 / legal.len() as f64;
            for &a in &legal {
                sigma[a as usize] = u;
            }
        }
    }

    if p == traverser {
        let mut util = [0f64; N_ACT];
        let mut value = 0f64;
        for &a in &legal {
            let mut child = hand.clone();
            child.step(a);
            util[a as usize] = traverse(&child, traverser, nodes, rng, cache, n_sims);
            value += sigma[a as usize] * util[a as usize];
        }
        let node = nodes.get_mut(&key).expect("nœud disparu");
        for &a in &legal {
            let i = a as usize;
            node[i] = ((node[i] as f64) + util[i] - value).max(0.0) as f32; // RM+
        }
        value
    } else {
        {
            let node = nodes.get_mut(&key).expect("nœud disparu");
            for &a in &legal {
                node[N_ACT + a as usize] += sigma[a as usize] as f32;
            }
        }
        let r = rng.f64();
        let mut acc = 0f64;
        let mut chosen = *legal.last().unwrap();
        for &a in &legal {
            acc += sigma[a as usize];
            if r < acc {
                chosen = a;
                break;
            }
        }
        let mut child = hand.clone();
        child.step(chosen);
        traverse(&child, traverser, nodes, rng, cache, n_sims)
    }
}

/// Une itération complète : une donne, un traverseur (alternés comme en Python :
/// bouton = (t/2) % 2, traverseur = t % 2, t = compteur d'itérations).
pub fn run_iteration(nodes: &mut Nodes, iterations: u64, rng: &mut Rng, n_sims: usize, version: u8) {
    let mut deck: Vec<u8> = (0..52).collect();
    rng.partial_shuffle(&mut deck, 9);
    let button = ((iterations / 2) % 2) as usize;
    let traverser = (iterations % 2) as usize;
    let hand = Hand::new(&deck, button);
    let mut cache = DealCache::new(version);
    traverse(&hand, traverser, nodes, rng, &mut cache, n_sims);
}

/// Fusionne les tables de `workers` sur `base` : final = Σ workers − (k−1)·base,
/// régrets ET stratégie replanchés à zéro (RM+), nœuds nuls élagués.
/// Reproduit exactement merge_workers() de train_cfr_parallel.py — mais en RAM,
/// sans jamais écrire de fichiers ouvriers sur le disque.
pub fn merge(base: &Nodes, workers: Vec<Nodes>) -> Nodes {
    let k = workers.len() as f32;
    let scale = -(k - 1.0);
    let mut acc: Nodes = Nodes::with_capacity(base.len());
    for (key, node) in base {
        let mut v = *node;
        for x in v.iter_mut() {
            *x *= scale;
        }
        acc.insert(key.clone(), v);
    }
    for w in workers {
        for (key, node) in w {
            match acc.get_mut(&key) {
                Some(e) => {
                    for i in 0..NODE {
                        e[i] += node[i];
                    }
                }
                None => {
                    acc.insert(key, node);
                }
            }
        }
    }
    acc.retain(|_, node| {
        for x in node.iter_mut() {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
        node.iter().any(|&x| x != 0.0) // élague les nœuds entièrement nuls
    });
    acc
}
