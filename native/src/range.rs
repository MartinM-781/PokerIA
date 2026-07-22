//! Outils de ranges : équité d'une main contre une distribution pondérée de
//! mains adverses, et calcul de buckets par lot (pour traquer la range
//! adverse candidate par candidate sans payer 1225 appels Python).

use crate::bucket::{card_bucket_v2_from_parts, preflop_class};
use crate::equity::equity_vs_random;
use crate::eval::evaluate7;
use crate::rng::Rng;

/// Équité de `hole` contre une range pondérée, board complété par tirage.
/// `pairs` : mains candidates (c1, c2), `weights` : poids (pas forcément
/// normalisés). Les candidats en conflit avec `hole`/`board` sont ignorés.
pub fn equity_vs_range(
    hole: &[u8; 2],
    board: &[u8],
    pairs: &[(u8, u8)],
    weights: &[f64],
    n_sims: usize,
    rng: &mut Rng,
) -> f64 {
    let mut blocked = [false; 52];
    blocked[hole[0] as usize] = true;
    blocked[hole[1] as usize] = true;
    for &c in board {
        blocked[c as usize] = true;
    }
    // Distribution cumulée des candidats valides
    let mut cum: Vec<(f64, usize)> = Vec::with_capacity(pairs.len());
    let mut total = 0.0;
    for (i, &(a, b)) in pairs.iter().enumerate() {
        if blocked[a as usize] || blocked[b as usize] || weights[i] <= 0.0 {
            continue;
        }
        total += weights[i];
        cum.push((total, i));
    }
    if cum.is_empty() {
        return 0.5;
    }

    let mut my7 = [0u8; 7];
    my7[0] = hole[0];
    my7[1] = hole[1];
    my7[2..2 + board.len()].copy_from_slice(board);
    let mut opp7 = [0u8; 7];
    opp7[2..2 + board.len()].copy_from_slice(board);

    let need = 5 - board.len();
    let mut wins = 0f64;
    for _ in 0..n_sims {
        // tire une main adverse selon les poids
        let r = rng.f64() * total;
        let idx = match cum.binary_search_by(|probe| probe.0.partial_cmp(&r).unwrap()) {
            Ok(i) => i,
            Err(i) => i.min(cum.len() - 1),
        };
        let (oa, ob) = pairs[cum[idx].1];
        opp7[0] = oa;
        opp7[1] = ob;
        // complète le board sans conflit
        let mut deck: Vec<u8> = (0..52u8)
            .filter(|&c| !blocked[c as usize] && c != oa && c != ob)
            .collect();
        rng.partial_shuffle(&mut deck, need);
        for i in 0..need {
            my7[2 + board.len() + i] = deck[i];
            opp7[2 + board.len() + i] = deck[i];
        }
        let ms = evaluate7(&my7);
        let os = evaluate7(&opp7);
        if ms > os {
            wins += 1.0;
        } else if ms == os {
            wins += 0.5;
        }
    }
    wins / n_sims as f64
}

/// Buckets v2/v3 d'un lot de mains candidates sur un board donné.
/// street : 0 préflop (classes exactes), 1..3 postflop (équité + texture + tirages).
pub fn buckets_batch(
    board: &[u8],
    street: u8,
    pairs: &[(u8, u8)],
    n_sims: usize,
    rng: &mut Rng,
) -> Vec<String> {
    let mut out = Vec::with_capacity(pairs.len());
    for &(a, b) in pairs {
        let hole = [a, b];
        if street == 0 {
            out.push(preflop_class(&hole));
        } else {
            let eq = equity_vs_random(&hole, board, n_sims, rng);
            out.push(card_bucket_v2_from_parts(&hole, board, street, eq));
        }
    }
    out
}
