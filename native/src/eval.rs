//! Évaluateur 7 cartes — port scalaire exact de poker_ai/evaluator.py.
//! Même encodage de score : (cat << 20) | (k1 << 16) | (k2 << 12) | (k3 << 8) | (k4 << 4) | k5
//! avec les rangs manquants bornés à 0. Les scores doivent être IDENTIQUES
//! à ceux du Python, pas seulement le même ordre.

const HIGH_CARD: u32 = 0;
const PAIR: u32 = 1;
const TWO_PAIR: u32 = 2;
const TRIPS: u32 = 3;
const STRAIGHT: u32 = 4;
const FLUSH: u32 = 5;
const FULL_HOUSE: u32 = 6;
const QUADS: u32 = 7;
const STRAIGHT_FLUSH: u32 = 8;

#[inline]
fn clamp0(x: i32) -> u32 {
    if x < 0 { 0 } else { x as u32 }
}

#[inline]
fn pack(cat: u32, k1: i32, k2: i32, k3: i32, k4: i32, k5: i32) -> u32 {
    (cat << 20)
        | (clamp0(k1) << 16)
        | (clamp0(k2) << 12)
        | (clamp0(k3) << 8)
        | (clamp0(k4) << 4)
        | clamp0(k5)
}

/// Rang haut de la meilleure suite dans `present` (13 booléens), roue incluse.
fn straight_high(present: &[bool; 13]) -> i32 {
    for h in (4..=12).rev() {
        if present[h] && present[h - 1] && present[h - 2] && present[h - 3] && present[h - 4] {
            return h as i32;
        }
    }
    if present[12] && present[0] && present[1] && present[2] && present[3] {
        return 3;
    }
    -1
}

/// Les `k` plus hauts rangs présents, en excluant `excl` (-1 = pas d'exclusion).
fn top_excluding(present: &[bool; 13], excl: &[i32], out: &mut [i32]) {
    let mut idx = 0;
    for r in (0..13).rev() {
        if present[r] && !excl.contains(&(r as i32)) {
            if idx < out.len() {
                out[idx] = r as i32;
                idx += 1;
            } else {
                return;
            }
        }
    }
    while idx < out.len() {
        out[idx] = -1;
        idx += 1;
    }
}

pub fn evaluate7(cards: &[u8; 7]) -> u32 {
    let mut rank_counts = [0u8; 13];
    let mut suit_counts = [0u8; 4];
    for &c in cards {
        rank_counts[(c / 4) as usize] += 1;
        suit_counts[(c % 4) as usize] += 1;
    }
    let mut present = [false; 13];
    for r in 0..13 {
        present[r] = rank_counts[r] > 0;
    }

    // Couleur
    let mut flush_suit: i32 = -1;
    for s in 0..4 {
        if suit_counts[s] >= 5 {
            flush_suit = s as i32;
            break;
        }
    }
    let mut flush_present = [false; 13];
    if flush_suit >= 0 {
        for &c in cards {
            if (c % 4) as i32 == flush_suit {
                flush_present[(c / 4) as usize] = true;
            }
        }
    }

    let straight = straight_high(&present);
    let sf = if flush_suit >= 0 { straight_high(&flush_present) } else { -1 };

    // Groupes
    let mut pairs: Vec<i32> = Vec::with_capacity(3);
    let mut trips: Vec<i32> = Vec::with_capacity(2);
    let mut quad: i32 = -1;
    for r in (0..13).rev() {
        match rank_counts[r] {
            2 => pairs.push(r as i32),
            3 => trips.push(r as i32),
            4 => quad = r as i32,
            _ => {}
        }
    }
    let p1 = *pairs.first().unwrap_or(&-1);
    let p2 = *pairs.get(1).unwrap_or(&-1);
    let t1 = *trips.first().unwrap_or(&-1);
    let t2 = *trips.get(1).unwrap_or(&-1);

    let has_full = trips.len() >= 2 || (trips.len() >= 1 && !pairs.is_empty());

    if sf >= 0 {
        return pack(STRAIGHT_FLUSH, sf, -1, -1, -1, -1);
    }
    if quad >= 0 {
        let mut k = [-1i32; 1];
        top_excluding(&present, &[quad], &mut k);
        return pack(QUADS, quad, k[0], -1, -1, -1);
    }
    if has_full {
        let full_pair = p1.max(t2);
        return pack(FULL_HOUSE, t1, full_pair, -1, -1, -1);
    }
    if flush_suit >= 0 {
        let mut k = [-1i32; 5];
        top_excluding(&flush_present, &[], &mut k);
        return pack(FLUSH, k[0], k[1], k[2], k[3], k[4]);
    }
    if straight >= 0 {
        return pack(STRAIGHT, straight, -1, -1, -1, -1);
    }
    if !trips.is_empty() {
        let mut k = [-1i32; 2];
        top_excluding(&present, &[t1], &mut k);
        return pack(TRIPS, t1, k[0], k[1], -1, -1);
    }
    if pairs.len() >= 2 {
        let mut k = [-1i32; 1];
        top_excluding(&present, &[p1, p2], &mut k);
        return pack(TWO_PAIR, p1, p2, k[0], -1, -1);
    }
    if !pairs.is_empty() {
        let mut k = [-1i32; 3];
        top_excluding(&present, &[p1], &mut k);
        return pack(PAIR, p1, k[0], k[1], k[2], -1);
    }
    let mut k = [-1i32; 5];
    top_excluding(&present, &[], &mut k);
    pack(HIGH_CARD, k[0], k[1], k[2], k[3], k[4])
}
