"""LBR (Local Best Response) — borne inférieure d'exploitabilité d'un blueprint.

L'agent LBR connaît la STRATÉGIE du blueprint (pas ses cartes). Il traque la
range adverse par Bayes, puis choisit à chaque décision l'action d'EV maximal :
- fold : perd sa mise égalisée ;
- check/call : « call-down » — équité contre la range, pot égalisé ;
- relance : la range adverse répond selon le blueprint (fold → on gagne le pot ;
  continue → équité contre la range qui continue, pot grossi).

Le score de LBR contre le blueprint (bb/100) est une borne INFÉRIEURE de son
exploitabilité : un vrai meilleur-réponse ferait au moins aussi bien.
Référence : Lisý & Bowling, « Equilibrium Approximation Quality of Current
No-Limit Poker Bots » (2017).

Usage :
    python lbr.py models/cfr_v2/cfr_blueprint_v2_final.pkl --hands 1000
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.environ.get("TEMP", ""), "claude",
    "C--Users-marti-Downloads", "830761da-e62c-419e-ae22-042a9a58a5e7",
    "scratchpad", "pnlib"))
try:
    import poker_native as native
except ImportError:
    native = None

from poker_ai import game
from poker_ai.cfr import ACTION_CHARS, CFRPolicy, NodeStore, matched_strategy
from poker_ai.game import HeadsUpHand
from poker_ai.ranges import HandRanges

FLOOR_SIMS = 400  # simulations d'équité par décision LBR


class LBRAgent:
    """Meilleure réponse locale contre un blueprint donné."""

    def __init__(self, store, seat, seed=0, restrict_to_blueprint_actions=True):
        self.store = store
        self.seat = seat
        self.seed = seed
        self.restrict = restrict_to_blueprint_actions
        self.ranges = None
        self._decision = 0

    def new_hand(self, hand):
        self.ranges = HandRanges(self.store, viewer=self.seat, hand=hand,
                                 native=native, seed=self.seed)

    def _opp_response(self, hand, my_action, pairs, weights):
        """(masse de fold, poids de la range qui continue) si je joue my_action."""
        opp = 1 - self.seat
        na = self.store.n_actions
        street = hand.street
        board = hand.board
        # historique + mon action
        hist = ""
        last_s = game.PREFLOP
        for s, _p, a in hand.history:
            if s != last_s:
                hist += "/"
                last_s = s
            hist += ACTION_CHARS[a]
        if street != last_s:
            hist += "/"
        hist += ACTION_CHARS[min(my_action, na - 1) if my_action >= na else my_action]
        pos = "B" if hand.button == opp else "N"
        buckets = self.ranges._buckets(opp, street, board)
        legal_mask = np.ones(na, dtype=bool)

        fold_mass = 0.0
        cont = weights.copy()
        for i in range(len(weights)):
            w = weights[i]
            if w <= 0.0:
                continue
            node = self.store.nodes.get(f"{pos}|{buckets[i]}|{hist}")
            if node is None:
                p_fold = 0.0  # inconnu : le blueprint réel replie sur check/call
            else:
                strat = node[na:]
                total = float(strat.sum())
                if total > 1e-9:
                    p_fold = float(strat[game.FOLD]) / total
                else:
                    p_fold = float(matched_strategy(node[:na], legal_mask)[game.FOLD])
            fold_mass += w * p_fold
            cont[i] = w * (1.0 - p_fold)
        return fold_mass, cont

    def act(self, hand, player):
        assert player == self.seat
        self.ranges.sync(hand)
        pairs, weights = self.ranges.normalized(1 - self.seat)
        me, opp = self.seat, 1 - self.seat
        hole = [int(c) for c in hand.hole[me]]
        board = [int(c) for c in hand.board]
        to_call = hand.to_call(me)
        legal = hand.legal_actions()
        if self.restrict:
            legal = [a for a in legal if a < self.store.n_actions]

        self._decision += 1
        seed = self.seed * 100_003 + self._decision
        plist = [(int(a), int(b)) for a, b in pairs]
        wlist = [float(x) for x in weights]

        best_a, best_ev = legal[0], -1e18
        for a in legal:
            if a == game.FOLD:
                ev = -float(min(hand.invested[me], hand.invested[opp]))
            elif a == game.CHECK_CALL:
                pay = min(to_call, hand.stacks[me])
                matched = min(hand.invested[me] + pay, hand.invested[opp])
                eq = native.equity_vs_range(hole, board, plist, wlist,
                                            FLOOR_SIMS, seed + a)
                ev = (2.0 * eq - 1.0) * matched
            else:
                # taille de la relance (mêmes règles que le moteur)
                pot_after_call = hand.pot + to_call
                if a == game.RAISE_THIRD:
                    raise_by = pot_after_call // 3
                elif a == game.RAISE_HALF:
                    raise_by = pot_after_call // 2
                elif a == game.RAISE_POT:
                    raise_by = pot_after_call
                else:
                    raise_by = hand.stacks[me]
                raise_by = max(raise_by, hand.last_raise, game.BB)
                add = min(to_call + raise_by, hand.stacks[me])
                my_inv = hand.invested[me] + add

                fold_mass, cont = self._opp_response(hand, a, pairs, weights)
                cont_total = float(np.sum(cont))
                win_now = float(hand.invested[opp])  # il se couche : je gagne sa mise
                if cont_total <= 1e-12:
                    ev = win_now
                else:
                    # il continue (au minimum il paie) : pot égalisé au niveau relancé
                    opp_inv_called = min(hand.invested[opp] + (my_inv - hand.invested[opp]),
                                         hand.invested[opp] + hand.stacks[opp])
                    matched = min(my_inv, opp_inv_called)
                    clist = [float(x) for x in cont]
                    eq = native.equity_vs_range(hole, board, plist, clist,
                                                FLOOR_SIMS, seed + 10 + a)
                    p_fold = fold_mass / (fold_mass + cont_total)
                    ev = p_fold * win_now + (1 - p_fold) * (2.0 * eq - 1.0) * matched
            if ev > best_ev:
                best_a, best_ev = a, ev
        return best_a


def measure(blueprint_path, n_hands, seed=0, restrict=True, log=print):
    store = NodeStore.load(blueprint_path)
    rng = np.random.default_rng(seed)
    results = np.zeros(n_hands)
    t0 = time.time()
    for i in range(n_hands):
        hand = HeadsUpHand(rng, button=i % 2)
        lbr = LBRAgent(store, seat=0, seed=seed + i, restrict_to_blueprint_actions=restrict)
        lbr.new_hand(hand)
        bp = CFRPolicy(store, rng, n_sims=160)
        while not hand.terminal:
            actor = lbr if hand.to_act == 0 else bp
            hand.step(actor.act(hand, hand.to_act))
        results[i] = hand.payoffs[0] / game.BB
        if (i + 1) % 200 == 0:
            bb = results[:i + 1]
            log(f"  {i + 1}/{n_hands} : LBR {bb.mean() * 100:+.0f} bb/100 "
                f"(±{bb.std() / np.sqrt(len(bb)) * 100:.0f}) "
                f"| {(i + 1) / (time.time() - t0):.1f} mains/s", flush=True)
    bb = results
    return bb.mean() * 100, bb.std() / np.sqrt(n_hands) * 100


def main():
    parser = argparse.ArgumentParser(description="Exploitabilité (borne inf.) par LBR")
    parser.add_argument("model")
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-actions", action="store_true",
                        help="LBR peut utiliser des actions hors du vocabulaire du blueprint")
    args = parser.parse_args()
    if native is None:
        print("poker_native requis")
        sys.exit(1)
    print(f"LBR contre {args.model} ({args.hands} mains)…", flush=True)
    bb, se = measure(args.model, args.hands, seed=args.seed,
                     restrict=not args.all_actions)
    print(f"\nExploitabilité (borne inférieure) : {bb:+.0f} bb/100 (±{se:.0f})", flush=True)


if __name__ == "__main__":
    main()
