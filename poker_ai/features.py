"""Transformation d'un état de jeu en vecteur de features pour le réseau.

v2 (27 features) : ajoute la texture du board (board pairé, tirages/possibilités
de couleur, coordination pour les suites) et la lecture de l'agression adverse
(qui relance, combien de fois, qui a l'initiative) — indispensables pour tenir
face à un adversaire humain qui bluffe et calibre ses mises.
"""
from collections import Counter

import numpy as np

from .equity import equity_vs_random
from .game import START_STACK

N_FEATURES = 27


def _board_texture(board):
    """(pairé, couleur possible ≥3, tirage couleur =2, board coordonné) en 0/1."""
    if not board:
        return 0.0, 0.0, 0.0, 0.0
    ranks = [c // 4 for c in board]
    suits = [c % 4 for c in board]
    paired = 1.0 if len(set(ranks)) < len(ranks) else 0.0
    max_suit = max(Counter(suits).values())
    flush_possible = 1.0 if max_suit >= 3 else 0.0
    flush_draw = 1.0 if max_suit == 2 else 0.0
    rank_set = set(ranks)
    if 12 in rank_set:  # l'as joue aussi en bas pour la roue
        rank_set.add(-1)
    coordinated = 0.0
    for low in range(-1, 9):  # une fenêtre de 5 rangs contenant ≥3 cartes du board
        if len(rank_set & {low, low + 1, low + 2, low + 3, low + 4}) >= 3:
            coordinated = 1.0
            break
    return paired, flush_possible, flush_draw, coordinated


def extract_features(hand, player, rng, n_sims=100):
    """Vecteur (N_FEATURES,) float32 décrivant la situation du point de vue de `player`."""
    p = player
    o = 1 - p
    pot = hand.pot
    to_call = min(max(hand.bets[o] - hand.bets[p], 0), hand.stacks[p])

    equity = equity_vs_random(hand.hole[p], hand.board, n_sims, rng)
    pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0
    spr = min(hand.stacks[p], hand.stacks[o]) / max(pot, 1)

    street_one_hot = [0.0, 0.0, 0.0, 0.0]
    street_one_hot[hand.street] = 1.0

    paired, flush_possible, flush_draw, coordinated = _board_texture(hand.board)

    return np.array([
        equity,
        equity * equity,
        *street_one_hot,
        1.0 if hand.button == p else 0.0,
        pot / (2 * START_STACK),
        pot_odds,
        to_call / START_STACK,
        hand.stacks[p] / START_STACK,
        hand.stacks[o] / START_STACK,
        hand.invested[p] / START_STACK,
        hand.invested[o] / START_STACK,
        min(spr, 20.0) / 20.0,
        min(hand.raises_this_street, 4) / 4.0,
        equity - pot_odds,
        # --- agression adverse et propre historique de relances ---
        min(hand.street_raise_counts[o], 4) / 4.0,
        min(hand.raise_counts[o], 8) / 8.0,
        min(hand.raise_counts[p], 8) / 8.0,
        1.0 if hand.last_aggressor == o else 0.0,
        # --- texture du board ---
        paired,
        flush_possible,
        flush_draw,
        coordinated,
        # --- pression sur les tapis ---
        to_call / max(hand.stacks[p] + to_call, 1),          # part du tapis à risquer
        # taille de mise adverse relative au pot, plafonnée à 3× (au-delà,
        # c'est « énorme » — sans plafond la valeur explosait à ~100 face aux all-ins)
        min(hand.bets[o] / max(pot - hand.bets[o], 1), 3.0) / 3.0 if hand.bets[o] else 0.0,
    ], dtype=np.float32)
