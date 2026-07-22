"""Équité (probabilité de gain) par simulation Monte-Carlo vectorisée."""
import numpy as np

from .evaluator import evaluate_hands


def equity_vs_random(hole, board, n_sims, rng):
    """Équité de `hole` (2 cartes) + `board` (0/3/4/5 cartes) contre une main
    adverse aléatoire, board complété aléatoirement. Renvoie un float 0..1."""
    known = list(hole) + list(board)
    known_set = set(known)
    deck = np.array([c for c in range(52) if c not in known_set], dtype=np.int64)

    need = 5 - len(board)
    k = 2 + need  # 2 cartes adverses + fin du board

    # Un tirage sans remise par simulation : k premiers indices d'une permutation
    perm = np.argsort(rng.random((n_sims, deck.size)), axis=1)[:, :k]
    draw = deck[perm]
    opp = draw[:, :2]
    runout = draw[:, 2:]

    mine = np.broadcast_to(np.array(known, dtype=np.int64), (n_sims, len(known)))
    my7 = np.column_stack([mine, runout])
    if board:
        board_arr = np.broadcast_to(np.array(board, dtype=np.int64), (n_sims, len(board)))
        opp7 = np.column_stack([opp, board_arr, runout])
    else:
        opp7 = np.column_stack([opp, runout])

    my_scores = evaluate_hands(my7)
    opp_scores = evaluate_hands(opp7)
    wins = (my_scores > opp_scores).sum()
    ties = (my_scores == opp_scores).sum()
    return float((wins + 0.5 * ties) / n_sims)
