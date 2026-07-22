"""Mesure la force de l'IA entraînée contre les bots de référence.

Usage :
    python evaluate.py                          # 2 000 mains contre chaque bot
    python evaluate.py --hands 5000 --opponent rule
"""
import argparse
import os

import numpy as np

from poker_ai import game
from poker_ai.agent import CallBot, NetworkPolicy, RandomBot, RuleBot
from poker_ai.game import HeadsUpHand
from poker_ai.network import QNetwork


def load_policy(model_path, rng, n_sims):
    """.pkl = blueprint CFR ; .npz = réseau DQN."""
    if model_path.endswith(".pkl"):
        from poker_ai.cfr import CFRPolicy
        return CFRPolicy(model_path, rng, n_sims=max(n_sims, 160))
    return NetworkPolicy(QNetwork.load(model_path), rng, eps=0.0, n_sims=n_sims)


def play_match(ai, opponent, n_hands, rng, n_sims):
    results = np.zeros(n_hands)
    for i in range(n_hands):
        hand = HeadsUpHand(rng, button=i % 2)
        while not hand.terminal:
            actor = ai if hand.to_act == 0 else opponent
            hand.step(actor.act(hand, hand.to_act))
        results[i] = hand.payoffs[0]
    bb = results / game.BB
    return bb.mean() * 100, bb.std() / np.sqrt(n_hands) * 100


def main():
    parser = argparse.ArgumentParser(description="Évaluation de l'IA de poker")
    parser.add_argument("--model", default="models/model.npz")
    parser.add_argument("--hands", type=int, default=2000, help="mains par adversaire")
    parser.add_argument("--opponent", choices=["random", "call", "rule", "dqn", "all"], default="all")
    parser.add_argument("--sims", type=int, default=200, help="simulations Monte-Carlo par décision")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ai = load_policy(args.model, rng, args.sims)
    opponents = {
        "random": ("Bot aléatoire", lambda: RandomBot(rng)),
        "call": ("Bot suiveur (calling station)", lambda: CallBot()),
        "rule": ("Bot à règles (équité + cote du pot)", lambda: RuleBot(rng, args.sims)),
    }
    dqn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_v3_dqn.npz")
    if os.path.exists(dqn_path) and args.model != dqn_path:
        opponents["dqn"] = ("IA DQN v3 (réseau de neurones)",
                            lambda: NetworkPolicy(QNetwork.load(dqn_path), rng,
                                                  eps=0.0, n_sims=args.sims))
    names = [args.opponent] if args.opponent != "all" else list(opponents)

    print(f"Modèle : {args.model} — {args.hands} mains par adversaire\n")
    for key in names:
        label, factory = opponents[key]
        bb100, stderr = play_match(ai, factory(), args.hands, rng, args.sims)
        verdict = "GAGNANT" if bb100 > 2 * stderr else ("PERDANT" if bb100 < -2 * stderr else "~égalité")
        print(f"  vs {label:<38} {bb100:+8.1f} bb/100  (±{stderr:.1f})  {verdict}")
    print("\n(bb/100 = grosses blinds gagnées pour 100 mains ; > 0 = l'IA gagne)")


if __name__ == "__main__":
    main()
