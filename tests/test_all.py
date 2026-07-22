"""Tests de validation : évaluateur, équité, moteur de jeu.

Lancer :  python tests/test_all.py
"""
import os
import sys
from collections import Counter
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_ai import game
from poker_ai.equity import equity_vs_random
from poker_ai.evaluator import evaluate_hand, evaluate_hands, hand_category
from poker_ai.game import HeadsUpHand


# ---------------------------------------------------------------------------
# Évaluateur naïf de référence (5 cartes, lent mais évident)
# ---------------------------------------------------------------------------

def eval5_naive(cards):
    ranks = sorted((c // 4 for c in cards), reverse=True)
    suits = [c % 4 for c in cards]
    counts = Counter(ranks)
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    ordered = [r for r, _ in groups]
    is_flush = len(set(suits)) == 1
    distinct = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(distinct) == 5:
        if distinct[0] - distinct[4] == 4:
            straight_high = distinct[0]
        elif distinct == [12, 3, 2, 1, 0]:  # roue A-2-3-4-5
            straight_high = 3
    sizes = sorted(counts.values(), reverse=True)
    if is_flush and straight_high is not None:
        return (8, straight_high)
    if sizes[0] == 4:
        return (7, ordered[0], ordered[1])
    if sizes == [3, 2]:
        return (6, ordered[0], ordered[1])
    if is_flush:
        return (5, *ranks)
    if straight_high is not None:
        return (4, straight_high)
    if sizes[0] == 3:
        return (3, ordered[0], *ordered[1:])
    if sizes[:2] == [2, 2]:
        return (2, ordered[0], ordered[1], ordered[2])
    if sizes[0] == 2:
        return (1, ordered[0], *ordered[1:])
    return (0, *ranks)


def eval7_naive(cards):
    return max(eval5_naive(c) for c in combinations(cards, 5))


def test_evaluator_vs_naive(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    for i in range(n):
        deck = rng.permutation(52)
        a, b = deck[:7].tolist(), deck[7:14].tolist()
        fa, fb = evaluate_hand(a), evaluate_hand(b)
        na, nb = eval7_naive(a), eval7_naive(b)
        assert hand_category(fa) == na[0], f"catégorie: {a} → {hand_category(fa)} vs {na[0]}"
        assert hand_category(fb) == nb[0], f"catégorie: {b} → {hand_category(fb)} vs {nb[0]}"
        fast_cmp = (fa > fb) - (fa < fb)
        naive_cmp = (na > nb) - (na < nb)
        assert fast_cmp == naive_cmp, f"ordre différent: {a} vs {b}"
    print(f"OK  évaluateur : {n} duels identiques à la référence naïve")


def test_evaluator_known_hands():
    def make(spec):  # spec: liste (rang, couleur)
        return [r * 4 + s for r, s in spec]

    royal = make([(12, 0), (11, 0), (10, 0), (9, 0), (8, 0), (0, 1), (1, 2)])
    quads = make([(12, 0), (12, 1), (12, 2), (12, 3), (8, 0), (0, 1), (1, 2)])
    full = make([(10, 0), (10, 1), (10, 2), (5, 0), (5, 1), (0, 2), (1, 3)])
    wheel = make([(12, 0), (0, 1), (1, 2), (2, 3), (3, 0), (7, 1), (9, 2)])
    assert hand_category(evaluate_hand(royal)) == 8
    assert hand_category(evaluate_hand(quads)) == 7
    assert hand_category(evaluate_hand(full)) == 6
    assert hand_category(evaluate_hand(wheel)) == 4
    assert evaluate_hand(royal) > evaluate_hand(quads) > evaluate_hand(full)
    print("OK  évaluateur : mains connues (quinte flush royale, carré, full, roue)")


def test_equity_sanity(seed=2):
    rng = np.random.default_rng(seed)
    aces = [12 * 4 + 0, 12 * 4 + 1]              # A♠ A♥
    seven_two = [5 * 4 + 0, 0 * 4 + 1]           # 7♠ 2♥ dépareillés
    eq_aa = equity_vs_random(aces, [], 4000, rng)
    eq_72 = equity_vs_random(seven_two, [], 4000, rng)
    assert 0.80 < eq_aa < 0.90, f"équité AA préflop suspecte : {eq_aa}"
    assert 0.28 < eq_72 < 0.45, f"équité 72o préflop suspecte : {eq_72}"
    # AA avec un troisième as au flop : quasi imbattable
    board = [12 * 4 + 2, 4 * 4 + 0, 7 * 4 + 1]
    eq_set = equity_vs_random(aces, board, 4000, rng)
    assert eq_set > 0.90, f"équité brelan d'as suspecte : {eq_set}"
    print(f"OK  équité : AA {eq_aa:.2f}, 72o {eq_72:.2f}, brelan d'as {eq_set:.2f}")


def test_game_invariants(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    max_steps = 0
    for i in range(n):
        hand = HeadsUpHand(rng, button=i % 2)
        steps = 0
        while not hand.terminal:
            legal = hand.legal_actions()
            assert legal, "aucune action légale"
            a = int(rng.choice(legal))
            hand.step(a)
            steps += 1
            assert steps < 200, "main sans fin"
            assert hand.stacks[0] >= 0 and hand.stacks[1] >= 0, "tapis négatif"
        max_steps = max(max_steps, steps)
        assert hand.payoffs[0] + hand.payoffs[1] == 0, "les jetons ne sont pas conservés"
        assert hand.stacks[0] + hand.stacks[1] == 2 * game.START_STACK
        assert abs(hand.payoffs[0]) <= game.START_STACK
    print(f"OK  moteur : {n} mains aléatoires, jetons conservés (max {max_steps} actions/main)")


def test_game_blinds_and_fold():
    rng = np.random.default_rng(4)
    hand = HeadsUpHand(rng, button=0)
    assert hand.invested == [game.SB, game.BB]
    assert hand.to_act == 0  # le bouton parle en premier préflop
    hand.step(game.FOLD)     # le bouton jette sa petite blind
    assert hand.terminal and hand.winner == 1
    assert hand.payoffs == [-game.SB, game.SB]
    print("OK  moteur : blinds et fold préflop")


def test_fold_refunds_uncalled_bet():
    """Quand on remporte un pot par fold après une mise non suivie, la part non
    payée est remboursée : le pot final = 2× la plus petite mise engagée."""
    rng = np.random.default_rng(5)
    hand = HeadsUpHand(rng, button=0)
    # Bouton (j0) part à tapis préflop, l'adversaire (j1) se couche.
    hand.step(game.ALL_IN)
    assert hand.to_act == 1 and not hand.terminal
    invested_before = list(hand.invested)
    hand.step(game.FOLD)
    assert hand.terminal and hand.winner == 0
    # Le gagnant ne remporte que ce que l'adversaire avait réellement engagé.
    matched = min(invested_before)
    assert hand.pot == 2 * matched, f"pot après fold = {hand.pot}, attendu {2 * matched}"
    assert hand.payoffs[0] == matched and hand.payoffs[1] == -matched
    assert hand.payoffs[0] + hand.payoffs[1] == 0
    print(f"OK  moteur : mise non suivie remboursée sur fold (pot réel {hand.pot})")


if __name__ == "__main__":
    test_evaluator_known_hands()
    test_evaluator_vs_naive()
    test_equity_sanity()
    test_game_blinds_and_fold()
    test_fold_refunds_uncalled_bet()
    test_game_invariants()
    print("\nTous les tests passent.")
