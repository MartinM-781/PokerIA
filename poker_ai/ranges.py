"""Suivi bayésien des ranges : quelles cartes chaque joueur peut-il avoir ?

Principe : chaque joueur est supposé jouer ~comme le blueprint. À chaque action
observée, le poids de chaque main candidate est multiplié par la probabilité
que le blueprint joue cette action avec cette main dans cette situation.
C'est la fondation commune de la recherche temps réel (ranges du résolveur)
et de la mesure d'exploitabilité LBR.
"""
import numpy as np

from . import game
from .cfr import ACTION_CHARS, matched_strategy

FLOOR = 0.02  # plancher : le blueprint mixe, aucune action n'exclut tout à fait une main


def all_pairs(excluded):
    """Toutes les paires de cartes possibles hors `excluded`."""
    ex = set(excluded)
    cards = [c for c in range(52) if c not in ex]
    return [(a, b) for i, a in enumerate(cards) for b in cards[i + 1:]]


class HandRanges:
    """Ranges des deux joueurs pour UNE main, mises à jour incrémentalement
    depuis `hand.history`. `viewer` est le siège qui observe (ses cartes sont
    retirées de la range adverse)."""

    def __init__(self, store, viewer, hand, native, seed=0):
        self.store = store
        self.viewer = viewer
        self.native = native
        self.seed = seed
        my_hole = list(hand.hole[viewer])
        # Range adverse : exclut mes cartes. Ma propre range (vue de l'adversaire) :
        # n'exclut que le board (il ne connaît pas mes cartes).
        self.pairs = {1 - viewer: all_pairs(my_hole), viewer: all_pairs([])}
        self.weights = {p: np.ones(len(self.pairs[p])) for p in (0, 1)}
        self._done = 0                # entrées d'historique déjà intégrées
        self._bucket_cache = {}       # (street, joueur) → liste de buckets alignée sur pairs
        self._hist_prefix = []        # chaîne d'historique reconstruite
        self._last_street = game.PREFLOP

    def _buckets(self, player, street, board):
        key = (street, player)
        got = self._bucket_cache.get(key)
        if got is None:
            got = self.native.buckets_batch(
                [int(c) for c in board], street,
                [(int(a), int(b)) for a, b in self.pairs[player]], 60,
                self.seed + street * 7 + player)
            self._bucket_cache[key] = got
        return got

    def _filter_board(self, board):
        """Retire les candidats en conflit avec les cartes du board visibles."""
        bd = set(board)
        for p in (0, 1):
            w = self.weights[p]
            for i, (a, b) in enumerate(self.pairs[p]):
                if a in bd or b in bd:
                    w[i] = 0.0

    def sync(self, hand):
        """Intègre les actions d'historique pas encore traitées."""
        history = hand.history
        while self._done < len(history):
            street, actor, action = history[self._done]
            # board tel qu'il était à cette street
            board = hand.full_board[:[0, 3, 4, 5][street]]
            if street != self._last_street:
                self._filter_board(board)
                self._last_street = street
            # clé d'infoset du candidat : position|bucket|historique_préfixe
            hist_str = ""
            last_s = game.PREFLOP
            for s, _p, a in self._hist_prefix:
                if s != last_s:
                    hist_str += "/"
                    last_s = s
                hist_str += ACTION_CHARS[a]
            pos = "B" if hand.button == actor else "N"
            buckets = self._buckets(actor, street, board)
            na = self.store.n_actions
            if action < na:  # une action inconnue du blueprint n'apprend rien
                w = self.weights[actor]
                nodes = self.store.nodes
                legal_mask = np.zeros(na, dtype=bool)
                # les actions légales exactes dépendent des jetons ; pour la
                # range on approxime par « toutes » — le plancher couvre l'écart
                legal_mask[:] = True
                for i in range(len(w)):
                    if w[i] <= 0.0:
                        continue
                    node = nodes.get(f"{pos}|{buckets[i]}|{hist_str}")
                    if node is None:
                        continue  # situation inconnue : poids inchangé
                    strat = node[na:]
                    total = float(strat.sum())
                    if total > 1e-9:
                        prob = float(strat[action]) / total
                    else:
                        prob = float(matched_strategy(node[:na], legal_mask)[action])
                    w[i] *= max(prob, FLOOR)
            self._hist_prefix.append((street, actor, action))
            self._done += 1
        # board courant (au cas où une street est arrivée sans action encore)
        if hand.street != self._last_street:
            self._filter_board(hand.board)
            self._last_street = hand.street

    def normalized(self, player):
        """(paires, poids normalisés) — poids nuls conservés (alignement)."""
        w = self.weights[player]
        total = w.sum()
        if total <= 0:
            w = np.ones_like(w)
            total = w.sum()
        return self.pairs[player], w / total
