"""Évaluation complète et précise d'un blueprint : force contre les bots de
référence, contre le DQN, et duel direct contre un autre blueprint.

Usage :
    python evaluate_full.py models/cfr_v2/cfr_blueprint.pkl \
        --vs models/cfr_blueprint_v1_final.pkl --hands 5000
"""
import argparse
import os

import numpy as np

from poker_ai import game
from poker_ai.agent import CallBot, NetworkPolicy, RandomBot, RuleBot
from poker_ai.cfr import CFRPolicy
from poker_ai.game import HeadsUpHand
from poker_ai.network import QNetwork

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def match(p0, p1, n, rng):
    res = np.zeros(n)
    for i in range(n):
        h = HeadsUpHand(rng, button=i % 2)
        while not h.terminal:
            actor = p0 if h.to_act == 0 else p1
            h.step(actor.act(h, h.to_act))
        res[i] = h.payoffs[0] / game.BB
    return res.mean() * 100, res.std() / np.sqrt(n) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--vs", default=None, help="blueprint adverse pour un duel direct")
    parser.add_argument("--hands", type=int, default=5000)
    parser.add_argument("--sims", type=int, default=160)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ai = CFRPolicy(args.model, rng, n_sims=args.sims)
    dqn_path = os.path.join(BASE_DIR, "models", "model_v3_dqn.npz")

    print(f"Évaluation : {args.model}  ({args.hands} mains/adversaire)\n", flush=True)
    rows = []
    for name, factory, n in [
        ("bot à règles", lambda: RuleBot(rng, 128), args.hands),
        ("bot suiveur", lambda: CallBot(), max(args.hands // 3, 800)),
        ("bot aléatoire", lambda: RandomBot(rng), max(args.hands // 3, 800)),
    ]:
        bb, se = match(ai, factory(), n, rng)
        rows.append((name, bb, se, n))
    if os.path.exists(dqn_path):
        dqn = NetworkPolicy(QNetwork.load(dqn_path), rng, eps=0.0, n_sims=128)
        bb, se = match(ai, dqn, max(args.hands // 2, 1500), rng)
        rows.append(("IA DQN v3", bb, se, max(args.hands // 2, 1500)))
    if args.vs and os.path.exists(args.vs):
        opp = CFRPolicy(args.vs, rng, n_sims=args.sims)
        bb, se = match(ai, opp, args.hands, rng)
        rows.append((f"duel vs {os.path.basename(args.vs)}", bb, se, args.hands))

    for name, bb, se, n in rows:
        verdict = "GAGNANT" if bb > 2 * se else ("PERDANT" if bb < -2 * se else "~égalité")
        print(f"  vs {name:<40} {bb:+8.1f} bb/100  (±{se:.0f}, {n} mains)  {verdict}", flush=True)
    print("\n(bb/100 > 0 = le blueprint gagne)", flush=True)


if __name__ == "__main__":
    main()
