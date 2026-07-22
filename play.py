"""Joue contre l'IA entraînée, dans le terminal.

Usage :
    python play.py                    # nécessite models/model.npz (lancer train.py d'abord)
    python play.py --sims 500         # IA plus précise (un peu plus lente)
"""
import argparse
import os
import sys

import numpy as np

from poker_ai import game
from poker_ai.agent import NetworkPolicy
from poker_ai.cards import cards_str
from poker_ai.evaluator import CATEGORY_NAMES, hand_category
from poker_ai.game import (ACTION_NAMES, ALL_IN, CHECK_CALL, FOLD, HeadsUpHand,
                           RAISE_HALF, RAISE_POT, RAISE_QUARTER, RAISE_THIRD,
                           STREET_NAMES)

HUMAN, AI = 0, 1
KEYS = {"f": FOLD, "c": CHECK_CALL, "q": RAISE_QUARTER, "t": RAISE_THIRD,
        "r": RAISE_HALF, "p": RAISE_POT, "a": ALL_IN}
KEY_LABELS = {FOLD: "[f]old", CHECK_CALL: "[c]heck/call", RAISE_QUARTER: "[q] ¼ pot",
              RAISE_THIRD: "[t] ⅓ pot", RAISE_HALF: "[r]aise ½ pot",
              RAISE_POT: "[p]ot raise", ALL_IN: "[a]ll-in"}


def ask_action(hand):
    legal = hand.legal_actions()
    to_call = hand.to_call(HUMAN)
    info = f"pot {hand.pot} | à payer {to_call} | ton tapis {hand.stacks[HUMAN]}"
    choices = "  ".join(KEY_LABELS[a] for a in legal) + "  [q]uitter"
    while True:
        raw = input(f"  ({info})\n  Ton action ? {choices} > ").strip().lower()
        if raw == "q":
            return None
        if raw in KEYS and KEYS[raw] in legal:
            return KEYS[raw]
        print("  Action invalide.")


def show_street(hand, shown):
    if hand.street != shown and hand.street > 0:
        print(f"\n  --- {STREET_NAMES[hand.street].upper()} : "
              f"{cards_str(hand.board, color=True)}  (pot {hand.pot}) ---")
    return hand.street


def play_hand(ai_policy, rng, button, session):
    hand = HeadsUpHand(rng, button=button)
    pos = "au bouton (tu parles en premier préflop)" if button == HUMAN else "en grosse blind"
    print(f"\n{'=' * 58}\n  NOUVELLE MAIN — tu es {pos}")
    print(f"  Tes cartes : {cards_str(hand.hole[HUMAN], color=True)}")
    shown = 0
    while not hand.terminal:
        shown = show_street(hand, shown)
        if hand.to_act == HUMAN:
            action = ask_action(hand)
            if action is None:
                return None
            hand.step(action)
        else:
            action = ai_policy.act(hand, AI)
            hand.step(action)
            print(f"  L'IA {ACTION_NAMES[action]}"
                  + (f" (mise totale {hand.bets[AI]})" if action >= RAISE_HALF and not hand.terminal else ""))

    delta_bb = hand.payoffs[HUMAN] / game.BB
    session.append(delta_bb)
    print()
    if hand.showdown:
        print(f"  --- ABATTAGE — board : {cards_str(hand.full_board, color=True)} ---")
        print(f"  Toi  : {cards_str(hand.hole[HUMAN], color=True)}  "
              f"({CATEGORY_NAMES[hand_category(hand.scores[HUMAN])]})")
        print(f"  L'IA : {cards_str(hand.hole[AI], color=True)}  "
              f"({CATEGORY_NAMES[hand_category(hand.scores[AI])]})")
        result = ["Partage.", "Tu gagnes", "L'IA gagne"][0 if hand.winner == -1 else (1 if hand.winner == HUMAN else 2)]
    else:
        result = "Tu gagnes" if hand.winner == HUMAN else "L'IA gagne"
    if hand.winner != -1:
        print(f"  {result} le pot de {hand.pot} jetons.")
    else:
        print("  Partage du pot.")
    print(f"  Main : {delta_bb:+.1f} BB | Session : {sum(session):+.1f} BB sur {len(session)} main(s)")
    return delta_bb


def main():
    if os.name == "nt":
        os.system("")  # active les codes couleur ANSI sous Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Jouer contre l'IA de poker")
    parser.add_argument("--model", default="models/model.npz")
    parser.add_argument("--sims", type=int, default=400,
                        help="simulations Monte-Carlo par décision de l'IA")
    parser.add_argument("--temperature", type=float, default=0.004,
                        help="stratégie mixte : 0 = déterministe, 0.004 = mélange les décisions serrées")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Modèle introuvable : {args.model}\nLance d'abord :  python train.py")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    from poker_ai.network import QNetwork
    ai_policy = NetworkPolicy(QNetwork.load(args.model), rng, eps=0.0,
                              n_sims=args.sims, temperature=args.temperature)

    print("=" * 58)
    print("  TEXAS HOLD'EM HEADS-UP — toi contre l'IA")
    print("  100 BB chacun à chaque main. Blinds 0,5/1 BB. [q] pour quitter.")
    session = []
    button = 0
    try:
        while True:
            if play_hand(ai_policy, rng, button, session) is None:
                break
            button = 1 - button
    except (KeyboardInterrupt, EOFError):
        pass
    if session:
        total = sum(session)
        print(f"\nFin de session : {total:+.1f} BB en {len(session)} main(s) "
              f"({total / len(session) * 100:+.0f} bb/100). "
              + ("Bien joué !" if total > 0 else "L'IA a été la plus forte cette fois."))


if __name__ == "__main__":
    main()
