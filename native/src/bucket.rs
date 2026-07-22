//! Abstraction v2 — clés d'infoset IDENTIQUES octet pour octet à
//! poker_ai/cfr.py (card_bucket_v2, _draw_flag, _board_texture, infoset_key).

use crate::equity::equity_vs_random;
use crate::game::{Hand, PREFLOP, RIVER};
use crate::rng::Rng;

const RANK_CHARS: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
const N_BUCKETS: usize = 12;
const N_BUCKETS_RIVER: usize = 16;
// fold, check/call, half, pot, all-in, third (indices d'action)
pub const ACTION_CHARS: [char; 6] = ['f', 'c', 'h', 'p', 'a', 't'];

/// Classe canonique préflop parmi 169 : "AA", "AKs", "72o"…
pub fn preflop_class(hole: &[u8; 2]) -> String {
    let mut r1 = (hole[0] / 4) as usize;
    let mut r2 = (hole[1] / 4) as usize;
    if r1 < r2 {
        std::mem::swap(&mut r1, &mut r2);
    }
    if r1 == r2 {
        let c = RANK_CHARS[r1];
        return format!("{}{}", c, c);
    }
    let suited = if hole[0] % 4 == hole[1] % 4 { 's' } else { 'o' };
    format!("{}{}{}", RANK_CHARS[r1], RANK_CHARS[r2], suited)
}

/// (board pairé, couleur possible ≥3) — comme _board_texture (board seul).
fn board_texture(board: &[u8]) -> (bool, bool) {
    if board.is_empty() {
        return (false, false);
    }
    let mut rank_seen = [false; 13];
    let mut paired = false;
    let mut suit_counts = [0u8; 4];
    for &c in board {
        let r = (c / 4) as usize;
        if rank_seen[r] {
            paired = true;
        }
        rank_seen[r] = true;
        suit_counts[(c % 4) as usize] += 1;
    }
    let flush_possible = suit_counts.iter().any(|&n| n >= 3);
    (paired, flush_possible)
}

/// Tirages du joueur : 2 = couleur, 1 = suite, 0 = rien — port de _draw_flag.
pub fn draw_flag(hole: &[u8; 2], board: &[u8]) -> u8 {
    let mut suit_counts = [0u8; 4];
    suit_counts[(hole[0] % 4) as usize] += 1;
    suit_counts[(hole[1] % 4) as usize] += 1;
    for &c in board {
        suit_counts[(c % 4) as usize] += 1;
    }
    for s in 0..4 {
        if suit_counts[s] == 4 && (hole[0] % 4 == s as u8 || hole[1] % 4 == s as u8) {
            return 2;
        }
    }
    // rangs avec l'as jouant aussi en bas (indice décalé de +1 : -1 → 0)
    let mut ranks = [false; 14];
    let mut hole_ranks = [false; 14];
    for &c in board {
        ranks[(c / 4 + 1) as usize] = true;
    }
    for &c in hole {
        ranks[(c / 4 + 1) as usize] = true;
        hole_ranks[(c / 4 + 1) as usize] = true;
    }
    if ranks[13] {
        ranks[0] = true;
    }
    if hole_ranks[13] {
        hole_ranks[0] = true;
    }
    for low in 0..=9usize {
        let mut n_present = 0;
        let mut hole_in = false;
        for i in low..low + 5 {
            if ranks[i] {
                n_present += 1;
            }
            if hole_ranks[i] {
                hole_in = true;
            }
        }
        if n_present == 4 && hole_in {
            return 1;
        }
    }
    0
}

/// Bucket v2 — chaîne identique à card_bucket_v2.
pub fn card_bucket_v2(hand: &Hand, player: usize, rng: &mut Rng, n_sims: usize) -> String {
    if hand.street == PREFLOP {
        return preflop_class(&hand.hole[player]);
    }
    let board = hand.board();
    let eq = equity_vs_random(&hand.hole[player], board, n_sims, rng);
    let (paired, flush_possible) = board_texture(board);
    let mut suffix = String::new();
    if paired {
        suffix.push('P');
    }
    if flush_possible {
        suffix.push('F');
    }
    if hand.street == RIVER {
        let b = ((eq * N_BUCKETS_RIVER as f64) as usize).min(N_BUCKETS_RIVER - 1);
        return format!("r{}{}", b, suffix);
    }
    let b = ((eq * N_BUCKETS as f64) as usize).min(N_BUCKETS - 1);
    let draw = draw_flag(&hand.hole[player], board);
    if draw > 0 {
        format!("{}{}D{}", b, suffix, draw)
    } else {
        format!("{}{}", b, suffix)
    }
}

/// Historique public : actions 'fchpa', streets séparées par '/'.
pub fn history_key(hand: &Hand) -> String {
    let mut out = String::with_capacity(hand.history.len() + 4);
    let mut last_street = PREFLOP;
    for &(street, _p, a) in &hand.history {
        if street != last_street {
            out.push('/');
            last_street = street;
        }
        out.push(ACTION_CHARS[a as usize]);
    }
    out
}

/// Clé d'infoset complète : "B|AKs|ch/hc" — identique au Python.
pub fn infoset_key(hand: &Hand, player: usize, bucket: &str) -> String {
    let pos = if hand.button == player { 'B' } else { 'N' };
    let mut key = String::with_capacity(bucket.len() + hand.history.len() + 8);
    key.push(pos);
    key.push('|');
    key.push_str(bucket);
    key.push('|');
    key.push_str(&history_key(hand));
    key
}
