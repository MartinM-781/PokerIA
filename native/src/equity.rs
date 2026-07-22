//! Équité Monte-Carlo contre main aléatoire — parité statistique avec
//! poker_ai/equity.py (même espérance, générateur différent).

use crate::eval::evaluate7;
use crate::rng::Rng;

pub fn equity_vs_random(hole: &[u8; 2], board: &[u8], n_sims: usize, rng: &mut Rng) -> f64 {
    let mut known = [false; 52];
    known[hole[0] as usize] = true;
    known[hole[1] as usize] = true;
    for &c in board {
        known[c as usize] = true;
    }
    let mut deck: Vec<u8> = (0..52u8).filter(|&c| !known[c as usize]).collect();

    let need = 5 - board.len();
    let k = 2 + need;

    let mut my7 = [0u8; 7];
    my7[0] = hole[0];
    my7[1] = hole[1];
    my7[2..2 + board.len()].copy_from_slice(board);

    let mut opp7 = [0u8; 7];
    opp7[2..2 + board.len()].copy_from_slice(board);

    let mut wins = 0u32;
    let mut ties = 0u32;
    for _ in 0..n_sims {
        rng.partial_shuffle(&mut deck, k);
        opp7[0] = deck[0];
        opp7[1] = deck[1];
        for i in 0..need {
            my7[2 + board.len() + i] = deck[2 + i];
            opp7[2 + board.len() + i] = deck[2 + i];
        }
        let ms = evaluate7(&my7);
        let os = evaluate7(&opp7);
        if ms > os {
            wins += 1;
        } else if ms == os {
            ties += 1;
        }
    }
    (wins as f64 + 0.5 * ties as f64) / n_sims as f64
}
