"""Évaluateur de mains de poker à 7 cartes, vectorisé avec NumPy.

`evaluate_hands` prend un tableau (N, 7) de cartes et renvoie un score entier
par ligne : plus le score est grand, plus la main est forte.

Encodage du score : (catégorie << 20) | (k1 << 16) | (k2 << 12) | (k3 << 8) | (k4 << 4) | k5
où les k sont les rangs départageant les égalités (kickers), du plus important
au moins important.
"""
import numpy as np

(HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH,
 FULL_HOUSE, QUADS, STRAIGHT_FLUSH) = range(9)

CATEGORY_NAMES = [
    "Hauteur", "Paire", "Deux paires", "Brelan", "Suite",
    "Couleur", "Full", "Carré", "Quinte flush",
]

_RANKS = np.arange(13)


def _top_ranks(present, k):
    """Pour chaque ligne, les k plus hauts rangs présents (rempli avec -1)."""
    vals = np.where(present, _RANKS[None, :], -1)
    return -np.sort(-vals, axis=1)[:, :k]


def _straight_high(present):
    """Rang haut de la meilleure suite par ligne, ou -1 (gère la roue A-2-3-4-5)."""
    n = present.shape[0]
    high = np.full(n, -1, dtype=np.int64)
    for h in range(12, 3, -1):
        window = present[:, h - 4:h + 1].all(axis=1)
        high = np.where((high < 0) & window, h, high)
    wheel = present[:, 12] & present[:, :4].all(axis=1)
    high = np.where((high < 0) & wheel, 3, high)
    return high


def evaluate_hands(cards):
    """cards : tableau (N, 7) d'entiers 0..51 → scores (N,) int64."""
    cards = np.asarray(cards, dtype=np.int64)
    n = cards.shape[0]
    ranks = cards // 4
    suits = cards % 4
    row = np.arange(n)[:, None]

    rank_counts = np.bincount((row * 13 + ranks).ravel(), minlength=n * 13).reshape(n, 13)
    suit_counts = np.bincount((row * 4 + suits).ravel(), minlength=n * 4).reshape(n, 4)
    present = rank_counts > 0

    # Couleur : présence des rangs dans la couleur majoritaire (>= 5 cartes)
    has_flush = (suit_counts >= 5).any(axis=1)
    flush_suit = np.where(has_flush, np.argmax(suit_counts >= 5, axis=1), -1)
    in_flush = suits == flush_suit[:, None]
    flush_ranks = np.where(in_flush, ranks, -1) + 1  # case 0 = poubelle
    flush_present = np.bincount(
        (row * 14 + flush_ranks).ravel(), minlength=n * 14
    ).reshape(n, 14)[:, 1:] > 0

    straight = _straight_high(present)
    straight_flush = _straight_high(flush_present)

    pairs = rank_counts == 2
    trips = rank_counts == 3
    quads = rank_counts == 4
    p2 = _top_ranks(pairs, 2)   # deux meilleures paires
    t2 = _top_ranks(trips, 2)   # deux meilleurs brelans
    q1 = _top_ranks(quads, 1)[:, 0]

    n_pairs = pairs.sum(axis=1)
    n_trips = trips.sum(axis=1)
    has_quads = quads.any(axis=1)
    has_full = (n_trips >= 2) | ((n_trips >= 1) & (n_pairs >= 1))
    has_trips = n_trips >= 1
    has_two_pair = n_pairs >= 2
    has_pair = n_pairs >= 1

    top5 = _top_ranks(present, 5)
    f5 = _top_ranks(flush_present, 5)

    def top_excluding(excludes, k):
        mask = present.copy()
        for e in excludes:
            mask &= _RANKS[None, :] != e[:, None]
        return _top_ranks(mask, k)

    quad_kick = top_excluding([q1], 1)[:, 0]
    full_pair = np.maximum(p2[:, 0], t2[:, 1])  # paire du full (ou 2e brelan)
    trips_kick = top_excluding([t2[:, 0]], 2)
    two_pair_kick = top_excluding([p2[:, 0], p2[:, 1]], 1)[:, 0]
    pair_kick = top_excluding([p2[:, 0]], 3)

    zeros = np.zeros(n, dtype=np.int64)

    def pack(cat, k1=None, k2=None, k3=None, k4=None, k5=None):
        def clamp(x):
            return zeros if x is None else np.maximum(np.asarray(x, dtype=np.int64), 0)
        return ((np.int64(cat) << 20) | (clamp(k1) << 16) | (clamp(k2) << 12)
                | (clamp(k3) << 8) | (clamp(k4) << 4) | clamp(k5))

    return np.select(
        [straight_flush >= 0,
         has_quads,
         has_full,
         has_flush,
         straight >= 0,
         has_trips,
         has_two_pair,
         has_pair],
        [pack(STRAIGHT_FLUSH, straight_flush),
         pack(QUADS, q1, quad_kick),
         pack(FULL_HOUSE, t2[:, 0], full_pair),
         pack(FLUSH, f5[:, 0], f5[:, 1], f5[:, 2], f5[:, 3], f5[:, 4]),
         pack(STRAIGHT, straight),
         pack(TRIPS, t2[:, 0], trips_kick[:, 0], trips_kick[:, 1]),
         pack(TWO_PAIR, p2[:, 0], p2[:, 1], two_pair_kick),
         pack(PAIR, p2[:, 0], pair_kick[:, 0], pair_kick[:, 1], pair_kick[:, 2])],
        default=pack(HIGH_CARD, top5[:, 0], top5[:, 1], top5[:, 2], top5[:, 3], top5[:, 4]),
    )


def evaluate_hand(cards7):
    """Score d'une seule main de 7 cartes (liste d'entiers)."""
    return int(evaluate_hands(np.asarray(cards7)[None, :])[0])


def hand_category(score):
    """Catégorie (0..8) d'un score."""
    return int(score) >> 20
