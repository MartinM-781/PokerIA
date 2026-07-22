"""MCCFR (Monte-Carlo Counterfactual Regret Minimization, external sampling).

Calcule une stratégie proche de l'équilibre de Nash de notre heads-up abstrait :
- abstraction de mises : les 5 actions du moteur (le jeu est déjà discret) ;
- abstraction de cartes : 169 classes préflop exactes ; postflop, buckets
  d'équité Monte-Carlo (12 niveaux) enrichis de la texture du board
  (board pairé, couleur possible).

Variante regret matching+ (régrets planchers à zéro), stratégie moyenne
accumulée sur les nœuds de l'adversaire échantillonné (Lanctot et al. 2009).
La stratégie finale est MIXTE par nature : elle ne peut pas être « lue »
par un humain, contrairement à une politique déterministe.
"""
import os
import pickle

import numpy as np

from . import game
from .cards import RANK_CHARS
from .equity import equity_vs_random
from .features import _board_texture

N_BUCKETS = 12          # buckets d'équité postflop (v1)
N_BUCKETS_RIVER = 16    # v2 : river plus fine (plus de tirages, que des mains faites)
LATEST_VERSION = 3      # v3 : ajoute les tailles ¼ et ⅓ pot
# fold, check/call, half-pot, pot, all-in, quarter-pot, third-pot
# (l'ordre suit les indices d'action du moteur)
ACTION_CHARS = "fchpaqt"


# ------------------------------------------------------------- abstraction

def preflop_class(hole):
    """Classe canonique préflop parmi 169 : 'AA', 'AKs', '72o'…"""
    r1, r2 = sorted((hole[0] // 4, hole[1] // 4), reverse=True)
    if r1 == r2:
        return RANK_CHARS[r1] * 2
    suited = "s" if hole[0] % 4 == hole[1] % 4 else "o"
    return RANK_CHARS[r1] + RANK_CHARS[r2] + suited


def card_bucket(hand, player, rng, n_sims=160):
    """Bucket v1 : équité seule + texture du board."""
    if hand.street == game.PREFLOP:
        return preflop_class(hand.hole[player])
    eq = equity_vs_random(hand.hole[player], hand.board, n_sims, rng)
    b = min(int(eq * N_BUCKETS), N_BUCKETS - 1)
    paired, flush_possible, _fd, _co = _board_texture(hand.board)
    suffix = ("P" if paired else "") + ("F" if flush_possible else "")
    return f"{b}{suffix}"


def _draw_flag(hole, board):
    """Tirages du joueur (déterministe, sans Monte-Carlo) :
    2 = tirage couleur, 1 = tirage suite (4 rangs d'une fenêtre de 5), 0 = rien.
    Le tirage doit impliquer au moins une carte fermée du joueur."""
    cards = list(hole) + list(board)
    suits = [c % 4 for c in cards]
    hole_suits = {c % 4 for c in hole}
    for s in range(4):
        if suits.count(s) == 4 and s in hole_suits:
            return 2
    ranks = {c // 4 for c in cards}
    hole_ranks = {c // 4 for c in hole}
    if 12 in ranks:
        ranks.add(-1)  # l'as joue en bas
    if 12 in hole_ranks:
        hole_ranks.add(-1)
    for low in range(-1, 9):
        window = {low, low + 1, low + 2, low + 3, low + 4}
        present = ranks & window
        if len(present) == 4 and hole_ranks & window:
            return 1
    return 0


def card_bucket_v2(hand, player, rng, n_sims=160):
    """Bucket v2 : équité + tirages (flop/turn) + texture, river plus fine.
    Une paire moyenne et un tirage couleur de même équité se jouent
    différemment — la v1 les confondait, la v2 les sépare."""
    if hand.street == game.PREFLOP:
        return preflop_class(hand.hole[player])
    eq = equity_vs_random(hand.hole[player], hand.board, n_sims, rng)
    paired, flush_possible, _fd, _co = _board_texture(hand.board)
    suffix = ("P" if paired else "") + ("F" if flush_possible else "")
    if hand.street == game.RIVER:  # plus de tirages : granularité d'équité accrue
        b = min(int(eq * N_BUCKETS_RIVER), N_BUCKETS_RIVER - 1)
        return f"r{b}{suffix}"
    b = min(int(eq * N_BUCKETS), N_BUCKETS - 1)
    draw = _draw_flag(hand.hole[player], hand.board)
    return f"{b}{suffix}D{draw}" if draw else f"{b}{suffix}"


# v3 partage l'abstraction de cartes de v2 (seul l'espace d'actions diffère).
BUCKET_FNS = {1: card_bucket, 2: card_bucket_v2, 3: card_bucket_v2}


def history_key(hand):
    """Séquence d'actions publiques, streets séparées par '/' : 'ch/hc/…'."""
    out = []
    last_street = game.PREFLOP
    for street, _p, a in hand.history:
        if street != last_street:
            out.append("/")
            last_street = street
        out.append(ACTION_CHARS[a])
    return "".join(out)


def infoset_key(hand, player, bucket):
    pos = "B" if hand.button == player else "N"  # bouton ou non
    return f"{pos}|{bucket}|{history_key(hand)}"


# ---------------------------------------------------------------- stockage
#
# Un nœud = UN tableau float32 de 2·N : [0:N] = régrets, [N:2N] = somme de
# stratégie, où N = nombre d'actions de l'abstraction (5 pour v1/v2, 7 pour v3).
# Un seul objet NumPy par situation : compact en RAM sur des millions de nœuds.

N_A = game.N_ACTIONS  # 7 (abstraction courante)


def n_actions_for(version):
    """Nombre d'actions de l'abstraction d'un blueprint (les indices 0..4 sont
    communs à toutes les versions ; v3 ajoute ¼ et ⅓ pot aux indices 5 et 6)."""
    return 7 if version >= 3 else 5


class NodeStore:
    """key → tableau (2·N,) float32 : régrets [0:N], stratégie moyenne [N:2N]."""

    def __init__(self, version=LATEST_VERSION):
        self.nodes = {}
        self.iterations = 0
        self.version = version  # version d'abstraction des buckets et des tailles
        self.n_actions = n_actions_for(version)

    def get(self, key):
        node = self.nodes.get(key)
        if node is None:
            node = np.zeros(2 * self.n_actions, dtype=np.float32)
            self.nodes[key] = node
        return node

    def save(self, path):
        # Sérialise la table telle quelle, sans copie (2× la RAM sinon).
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"iterations": self.iterations, "nodes": self.nodes,
                         "version": self.version, "format": 2,
                         "n_actions": self.n_actions},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls(version=data.get("version", 1))  # anciens fichiers = v1
        store.iterations = data["iterations"]
        if data.get("format", 1) == 2:
            store.nodes = data["nodes"]
        else:  # migration : ancien format [régrets, stratégie] séparés
            store.nodes = {k: np.concatenate([np.asarray(r, dtype=np.float32),
                                              np.asarray(s, dtype=np.float32)])
                           for k, (r, s) in data["nodes"].items()}
        # Le nombre d'actions est déduit du fichier, ou de la taille des nœuds
        # (compatibilité avec les blueprints v1/v2 à 5 actions).
        if "n_actions" in data:
            store.n_actions = data["n_actions"]
        elif store.nodes:
            store.n_actions = len(next(iter(store.nodes.values()))) // 2
        return store


# ------------------------------------------------------------------- MCCFR

def matched_strategy(regret, mask):
    """Regret matching : probabilités ∝ régrets positifs (uniforme sinon).
    Calcul en float64 : rng.choice exige une somme exactement égale à 1."""
    pos = np.where(mask, np.maximum(regret.astype(np.float64), 0.0), 0.0)
    total = pos.sum()
    if total > 1e-12:
        return pos / total
    uniform = mask.astype(np.float64)
    return uniform / uniform.sum()


class _DealCache:
    """Buckets par (joueur, street) pour une donne — l'équité coûte cher."""

    def __init__(self, rng, n_sims, version=LATEST_VERSION):
        self.rng = rng
        self.n_sims = n_sims
        self.bucket_fn = BUCKET_FNS[version]
        self.buckets = {}

    def bucket(self, hand, player):
        k = (player, hand.street)
        b = self.buckets.get(k)
        if b is None:
            b = self.bucket_fn(hand, player, self.rng, self.n_sims)
            self.buckets[k] = b
        return b


def traverse(hand, traverser, store, rng, cache):
    """External sampling : toutes les actions du traverseur, une seule
    (échantillonnée) pour l'adversaire. Renvoie l'utilité en BB."""
    if hand.terminal:
        return hand.payoffs[traverser] / game.BB

    p = hand.to_act
    legal = hand.legal_actions()
    mask = np.zeros(game.N_ACTIONS, dtype=bool)
    mask[legal] = True
    node = store.get(infoset_key(hand, p, cache.bucket(hand, p)))
    regret = node[:N_A]   # vues sur le tableau unique
    strat = node[N_A:]
    sigma = matched_strategy(regret, mask)

    if p == traverser:
        util = np.zeros(game.N_ACTIONS)
        for a in legal:
            child = hand.clone()
            child.step(a)
            util[a] = traverse(child, traverser, store, rng, cache)
        value = float((sigma * util).sum())
        regret[mask] += util[mask] - value
        np.maximum(regret, 0.0, out=regret)  # regret matching+
        return value

    strat[mask] += sigma[mask]  # stratégie moyenne (nœud échantillonné)
    action = int(rng.choice(game.N_ACTIONS, p=sigma))
    child = hand.clone()
    child.step(action)
    return traverse(child, traverser, store, rng, cache)


def run_iteration(store, rng, n_sims=160):
    """Une itération : une donne, un traverseur (alternés)."""
    t = store.iterations
    hand = game.HeadsUpHand(rng, button=(t // 2) % 2)
    cache = _DealCache(rng, n_sims, version=store.version)
    traverse(hand, t % 2, store, rng, cache)
    store.iterations += 1


# ----------------------------------------------------------------- en jeu

class CFRPolicy:
    """Joue la stratégie moyenne d'un blueprint MCCFR (échantillonnage mixte)."""

    def __init__(self, store_or_path, rng, n_sims=160):
        if isinstance(store_or_path, NodeStore):
            self.store = store_or_path
        else:
            self.store = NodeStore.load(store_or_path)
        self.rng = rng
        self.n_sims = n_sims
        self.bucket_fn = BUCKET_FNS[self.store.version]

    def _neighbor_buckets(self, bucket):
        """Buckets « voisins » par équité (±1, ±2) — même street, mêmes drapeaux.
        Sert de repli quand la situation exacte n'a jamais été explorée."""
        prefix = ""
        if bucket and bucket[0] == "r":
            prefix, bucket = "r", bucket[1:]
        digits = ""
        for ch in bucket:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            return []
        suffix = bucket[len(digits):]
        b = int(digits)
        return [f"{prefix}{b + d}{suffix}" for d in (-1, 1, -2, 2) if b + d >= 0]

    def act(self, hand, player):
        na = self.store.n_actions  # 5 pour un blueprint v1/v2, 7 pour v3
        legal = [a for a in hand.legal_actions() if a < na]
        mask = np.zeros(na, dtype=bool)
        mask[legal] = True
        bucket = self.bucket_fn(hand, player, self.rng, self.n_sims)
        node = self.store.nodes.get(infoset_key(hand, player, bucket))
        if node is None and hand.street != game.PREFLOP:
            for nb in self._neighbor_buckets(bucket):
                node = self.store.nodes.get(infoset_key(hand, player, nb))
                if node is not None:
                    break
        if node is None:  # vraiment jamais rien vu d'approchant : repli prudent
            return game.CHECK_CALL
        probs = np.where(mask, node[na:].astype(np.float64), 0.0)
        total = probs.sum()
        if total <= 1e-9:  # pas encore de stratégie moyenne : regret matching
            probs = matched_strategy(node[:na], mask)
        else:
            probs = probs / total
        return int(self.rng.choice(na, p=probs))
