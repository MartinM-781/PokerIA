"""Recherche en temps réel : au lieu de jouer la réponse pré-calculée du
blueprint, on RE-RÉSOUT la street courante au moment de décider.

À chaque décision postflop :
1. les ranges des deux joueurs sont traquées par Bayes sur le blueprint ;
2. le sous-jeu (enchères restantes de la street) est résolu par CFR
   range-contre-range côté Rust, feuilles évaluées par équité réelle ;
3. on joue la stratégie résolue pour NOTRE main réelle (mixte par nature).

Préflop, le blueprint est déjà exact (169 classes) : on le joue tel quel.
En cas de pépin (ranges dégénérées, module natif absent), repli blueprint.

C'est le principe qui sépare les bots « corrects » des bots de niveau
championnat (Libratus, Pluribus) — version d'une street, sans garanties de
sûreté théoriques, mais avec des ranges et une équité exactes.
"""
import numpy as np

from . import game
from .cfr import CFRPolicy, NodeStore
from .ranges import HandRanges

TOP_K = 250  # mains conservées par range (les plus probables) pour la vitesse


class SearchPolicy:
    """Politique = blueprint préflop + résolution temps réel postflop."""

    def __init__(self, store_or_path, rng, native, n_sims=200,
                 iterations=400, n_runouts=100):
        self.store = (store_or_path if isinstance(store_or_path, NodeStore)
                      else NodeStore.load(store_or_path))
        self.rng = rng
        self.native = native
        self.blueprint = CFRPolicy(self.store, rng, n_sims=n_sims)
        self.iterations = iterations
        self.n_runouts = n_runouts
        self.ranges = None
        self._hand_sig = None
        self._decision = 0

    # ------------------------------------------------------------------ range

    def _sync_ranges(self, hand, player):
        sig = (tuple(hand.hole[player]), tuple(hand.full_board))
        if sig != self._hand_sig:
            self.ranges = HandRanges(self.store, viewer=player, hand=hand,
                                     native=self.native,
                                     seed=int(self.rng.integers(1 << 30)))
            self._hand_sig = sig
        self.ranges.sync(hand)

    @staticmethod
    def _trim(pairs, weights, keep, must_include=None):
        """Garde les `keep` mains les plus probables (et la main imposée)."""
        order = np.argsort(weights)[::-1][:keep]
        chosen = list(order)
        if must_include is not None:
            mi = tuple(sorted(must_include))
            found = any(tuple(sorted(pairs[i])) == mi for i in chosen)
            if not found:
                for i, p in enumerate(pairs):
                    if tuple(sorted(p)) == mi:
                        chosen[-1] = i
                        break
        out_pairs = [(int(pairs[i][0]), int(pairs[i][1])) for i in chosen]
        out_w = np.maximum(weights[chosen], 1e-6)
        out_w = out_w / out_w.sum()
        return out_pairs, [float(x) for x in out_w]

    # ------------------------------------------------------------------- jeu

    def act(self, hand, player):
        if self.native is None or hand.street == game.PREFLOP:
            return self.blueprint.act(hand, player)
        try:
            return self._act_search(hand, player)
        except Exception:
            return self.blueprint.act(hand, player)  # repli robuste

    def _act_search(self, hand, player):
        self._sync_ranges(hand, player)
        opp = 1 - player
        pairs_o, w_o = self.ranges.normalized(opp)
        pairs_h, w_h = self.ranges.normalized(player)
        my_hole = (int(hand.hole[player][0]), int(hand.hole[player][1]))

        opp_pairs, opp_w = self._trim(pairs_o, w_o, TOP_K)
        hero_pairs, hero_w = self._trim(pairs_h, w_h, TOP_K, must_include=my_hole)

        self._decision += 1
        strat = self.native.solve_street(
            [int(c) for c in hand.board],
            hero_pairs, hero_w, opp_pairs, opp_w,
            (int(hand.invested[player]), int(hand.invested[opp])),
            (int(hand.bets[player]), int(hand.bets[opp])),
            (int(hand.stacks[player]), int(hand.stacks[opp])),
            int(hand.last_raise), int(hand.raises_this_street),
            (bool(hand.acted[player]), bool(hand.acted[opp])),
            True,
            self.iterations, self.n_runouts,
            int(self.rng.integers(1 << 60)),
        )
        my_sig = tuple(sorted(my_hole))
        idx = next(i for i, p in enumerate(hero_pairs)
                   if tuple(sorted(p)) == my_sig)
        probs = np.asarray(strat[idx], dtype=np.float64)

        legal = hand.legal_actions()
        mask = np.zeros(game.N_ACTIONS, dtype=bool)
        mask[legal] = True
        probs = np.where(mask[:len(probs)], probs, 0.0)
        total = probs.sum()
        if total <= 1e-9:
            return self.blueprint.act(hand, player)
        probs = probs / total
        return int(self.rng.choice(len(probs), p=probs))
