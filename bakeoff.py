"""Bake-off final : le nouveau modèle contre chaque candidat + le bot à règles.
Charge le nouveau une fois, puis chaque adversaire tour à tour (RAM maîtrisée).
"""
import gc
import sys
import time

import numpy as np

from poker_ai import game
from poker_ai.agent import RuleBot
from poker_ai.cfr import CFRPolicy, NodeStore
from poker_ai.game import HeadsUpHand

NEW = r"models/cfr_v3/cfr_blueprint.pkl"          # nuit : 27M post-purge
CANDIDATES = [
    ("v3@24M (pré-purge)", r"models/cfr_v3/cfr_blueprint_24M_prepurge.pkl"),
    ("v3@12M (servi actuel)", r"models/cfr_v3/cfr_blueprint_v3_final.pkl"),
    ("v2 finale", r"models/cfr_v2/cfr_blueprint_v2_final.pkl"),
]
HANDS = 3000


def duel(p0, p1, n, rng):
    res = np.zeros(n)
    for i in range(n):
        h = HeadsUpHand(rng, button=i % 2)
        while not h.terminal:
            actor = p0 if h.to_act == 0 else p1
            h.step(actor.act(h, h.to_act))
        res[i] = h.payoffs[0] / game.BB
    return res.mean() * 100, res.std() / np.sqrt(n) * 100


def main():
    rng = np.random.default_rng(2027)
    print(f"Bake-off : {NEW} (nuit) contre chaque candidat, {HANDS} mains\n", flush=True)
    new = CFRPolicy(NEW, rng, n_sims=160)

    bb, se = duel(new, RuleBot(rng, 128), HANDS, rng)
    verdict = "GAGNANT" if bb > 2 * se else ("PERDANT" if bb < -2 * se else "~nul")
    print(f"  vs {'bot à règles':<26} {bb:+8.1f} bb/100 (±{se:.0f})  {verdict}", flush=True)

    for label, path in CANDIDATES:
        opp = CFRPolicy(path, rng, n_sims=160)
        t0 = time.time()
        bb, se = duel(new, opp, HANDS, rng)
        verdict = "GAGNANT" if bb > 2 * se else ("PERDANT" if bb < -2 * se else "~nul")
        print(f"  vs {label:<26} {bb:+8.1f} bb/100 (±{se:.0f})  {verdict}  ({time.time()-t0:.0f}s)", flush=True)
        del opp
        gc.collect()

    print("\n(bb/100 > 0 = le modèle de la nuit gagne)", flush=True)


if __name__ == "__main__":
    main()
