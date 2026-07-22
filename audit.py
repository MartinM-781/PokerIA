"""Mode coach : révision de mains du bot, cartes visibles.

Le bot joue des mains contre un adversaire, chaque décision est enregistrée
avec ses cartes fermées, son équité et le contexte. Des règles signalent les
coups suspects, et les statistiques de style mesurent notamment s'il joue
différemment ses tirages et ses mains faites (l'aveuglement corrigé en v2).

Usage :
    python audit.py                                  # blueprint courant, 400 mains
    python audit.py --model models/cfr_blueprint_v1_final.pkl --hands 600
"""
import argparse
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from poker_ai import game
from poker_ai.agent import ManiacBot, RuleBot
from poker_ai.cards import cards_str
from poker_ai.cfr import CFRPolicy, _draw_flag
from poker_ai.equity import equity_vs_random
from poker_ai.game import ACTION_NAMES, HeadsUpHand

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI = 0


def audit(model_path, n_hands, seed=7, n_sims=300):
    rng = np.random.default_rng(seed)
    ai = CFRPolicy(model_path, rng, n_sims=200)
    opponents = [RuleBot(rng, 128), ManiacBot(rng)]
    decisions = []
    results = []

    for i in range(n_hands):
        hand = HeadsUpHand(rng, button=i % 2)
        opp = opponents[i % len(opponents)]
        while not hand.terminal:
            if hand.to_act == AI:
                equity = equity_vs_random(hand.hole[AI], hand.board, n_sims, rng)
                to_call = hand.to_call(AI)
                pot_odds = to_call / (hand.pot + to_call) if to_call else 0.0
                action = ai.act(hand, AI)
                decisions.append({
                    "hand_no": i,
                    "street": hand.street,
                    "hole": list(hand.hole[AI]),
                    "board": list(hand.board),
                    "equity": equity,
                    "pot_odds": pot_odds,
                    "to_call_bb": to_call / game.BB,
                    "pot_bb": hand.pot / game.BB,
                    "draw": _draw_flag(hand.hole[AI], hand.board) if hand.street in (1, 2) else 0,
                    "action": action,
                })
                hand.step(action)
            else:
                hand.step(opp.act(hand, 1))
        results.append(hand.payoffs[AI] / game.BB)

    return decisions, np.array(results)


def report(decisions, results):
    print(f"{len(results)} mains, {len(decisions)} décisions du bot "
          f"| résultat {results.mean() * 100:+.0f} bb/100\n")

    # ---- coups suspects
    flags = []
    for d in decisions:
        big_call = (d["action"] == game.CHECK_CALL and d["to_call_bb"] >= 8
                    and d["equity"] < d["pot_odds"] - 0.15)
        fold_monster = d["action"] == game.FOLD and d["equity"] > 0.70
        river_bluff = (d["street"] == game.RIVER and d["action"] >= game.RAISE_HALF
                       and d["equity"] < 0.20)
        if big_call:
            flags.append(("CALL cher sans cote", d))
        elif fold_monster:
            flags.append(("FOLD d'un monstre", d))
        elif river_bluff:
            flags.append(("Bluff river", d))

    print(f"Coups signalés : {len(flags)} / {len(decisions)} "
          f"({len(flags) / max(len(decisions), 1) * 100:.1f} %)")
    for label, d in flags[:12]:
        print(f"  [{label}] main n°{d['hand_no']} {game.STREET_NAMES[d['street']]} — "
              f"{cards_str(d['hole'])} | board {cards_str(d['board']) or '—'} | "
              f"équité {d['equity']:.2f} vs cote {d['pot_odds']:.2f} | "
              f"pot {d['pot_bb']:.0f} BB, à payer {d['to_call_bb']:.0f} BB → "
              f"{ACTION_NAMES[d['action']]}")

    # ---- style : tirages vs mains faites (à équité comparable, flop/turn)
    zone = [d for d in decisions if d["street"] in (1, 2) and 0.30 <= d["equity"] <= 0.55]
    draws = [d for d in zone if d["draw"] > 0]
    made = [d for d in zone if d["draw"] == 0]

    def aggr(group):
        return (sum(1 for d in group if d["action"] >= game.RAISE_HALF)
                / max(len(group), 1) * 100)

    print(f"\nStyle flop/turn à équité moyenne (0.30-0.55) :")
    print(f"  agressivité avec TIRAGE     : {aggr(draws):5.1f} %  ({len(draws)} décisions)")
    print(f"  agressivité MAIN FAITE      : {aggr(made):5.1f} %  ({len(made)} décisions)")
    print(f"  (un bon joueur semi-bluffe ses tirages : écart attendu chez la v2, pas la v1)")

    folds = sum(1 for d in decisions if d["action"] == game.FOLD)
    raises = sum(1 for d in decisions if d["action"] >= game.RAISE_HALF)
    print(f"\nProfil global : {folds / len(decisions) * 100:.0f} % fold, "
          f"{raises / len(decisions) * 100:.0f} % relance/mise")


def main():
    parser = argparse.ArgumentParser(description="Révision de mains du bot (mode coach)")
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "models", "cfr_blueprint.pkl"))
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(f"Audit de {args.model} sur {args.hands} mains (cartes visibles)…\n")
    decisions, results = audit(args.model, args.hands, seed=args.seed)
    report(decisions, results)


if __name__ == "__main__":
    main()
