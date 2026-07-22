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
                    "button": hand.button == AI,
                    "facing_bet": to_call > 0,
                    "was_aggressor": hand.last_aggressor == AI,
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


def _tier(equity_preflop):
    """Force préflop en 4 niveaux d'après l'équité contre main aléatoire."""
    if equity_preflop >= 0.60:
        return "premium"
    if equity_preflop >= 0.50:
        return "bonne"
    if equity_preflop >= 0.42:
        return "marginale"
    return "faible"


def build_aggregates(decisions, results):
    """Statistiques de style agrégées — le dossier que lisent les coachs."""
    def rate(group, pred):
        return round(sum(1 for d in group if pred(d)) / max(len(group), 1) * 100, 1)

    def action_mix(group):
        return {"n": len(group),
                "fold_%": rate(group, lambda d: d["action"] == game.FOLD),
                "call_%": rate(group, lambda d: d["action"] == game.CHECK_CALL),
                "raise_%": rate(group, lambda d: d["action"] >= game.RAISE_HALF)}

    pre = [d for d in decisions if d["street"] == game.PREFLOP]
    agg = {"resultat_bb100": round(float(np.mean(results)) * 100, 1),
           "mains": len(results), "decisions": len(decisions)}

    # Préflop par position et force de main
    agg["preflop"] = {}
    for pos, label in [(True, "bouton"), (False, "big_blind")]:
        group = [d for d in pre if d["button"] == pos]
        agg["preflop"][label] = {t: action_mix([d for d in group
                                                if _tier(d["equity"]) == t])
                                 for t in ("premium", "bonne", "marginale", "faible")}

    # Postflop : tirages vs mains faites à équité comparable (0.30-0.55)
    zone = [d for d in decisions if d["street"] in (1, 2) and 0.30 <= d["equity"] <= 0.55]
    agg["semi_bluff"] = {
        "avec_tirage": action_mix([d for d in zone if d["draw"] > 0]),
        "main_faite": action_mix([d for d in zone if d["draw"] == 0]),
    }

    # Défense face aux mises, par street
    agg["face_a_une_mise"] = {}
    for st in (1, 2, 3):
        group = [d for d in decisions if d["street"] == st and d["facing_bet"]]
        agg["face_a_une_mise"][game.STREET_NAMES[st]] = action_mix(group)

    # River : équilibre value/bluff quand le bot mise ou relance
    river_bets = [d for d in decisions
                  if d["street"] == game.RIVER and d["action"] >= game.RAISE_HALF]
    agg["river_agression"] = {
        "n": len(river_bets),
        "value_%": rate(river_bets, lambda d: d["equity"] >= 0.60),
        "bluff_%": rate(river_bets, lambda d: d["equity"] < 0.35),
        "entre_deux_%": rate(river_bets, lambda d: 0.35 <= d["equity"] < 0.60),
    }

    # Continuation bet : agresseur préflop qui mise au flop
    flop_as_aggressor = [d for d in decisions
                         if d["street"] == game.FLOP and d["was_aggressor"]
                         and not d["facing_bet"]]
    agg["cbet_flop_%"] = rate(flop_as_aggressor, lambda d: d["action"] >= game.RAISE_HALF)
    agg["cbet_flop_n"] = len(flop_as_aggressor)
    return agg


def flagged_hands(decisions):
    out = []
    for d in decisions:
        label = None
        if (d["action"] == game.CHECK_CALL and d["to_call_bb"] >= 8
                and d["equity"] < d["pot_odds"] - 0.15):
            label = "call_cher_sans_cote"
        elif d["action"] == game.FOLD and d["equity"] > 0.70:
            label = "fold_de_monstre"
        elif (d["street"] == game.RIVER and d["action"] >= game.RAISE_HALF
                and d["equity"] < 0.20):
            label = "bluff_river"
        if label:
            out.append({"type": label,
                        "street": game.STREET_NAMES[d["street"]],
                        "cartes": cards_str(d["hole"]),
                        "board": cards_str(d["board"]) or "-",
                        "equite": round(d["equity"], 2),
                        "cote_du_pot": round(d["pot_odds"], 2),
                        "pot_bb": round(d["pot_bb"], 1),
                        "a_payer_bb": round(d["to_call_bb"], 1),
                        "action": ACTION_NAMES[d["action"]]})
    return out


def main():
    parser = argparse.ArgumentParser(description="Révision de mains du bot (mode coach)")
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "models", "cfr_blueprint.pkl"))
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", default=None, metavar="FICHIER",
                        help="exporte le dossier complet (agrégats + mains signalées) en JSON")
    args = parser.parse_args()
    print(f"Audit de {args.model} sur {args.hands} mains (cartes visibles)…\n")
    decisions, results = audit(args.model, args.hands, seed=args.seed)
    report(decisions, results)
    if args.json:
        import json
        payload = {"modele": args.model,
                   "agregats": build_aggregates(decisions, results),
                   "mains_signalees": flagged_hands(decisions)}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"\nDossier coach exporté : {args.json}")


if __name__ == "__main__":
    main()
