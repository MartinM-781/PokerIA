"""Agent DQN (Deep Q-Learning) et bots de référence.

Le principe : à chaque décision, le réseau estime la valeur (en centaines de
BB) de chaque action. On apprend par différence temporelle sur les mains
jouées, avec un buffer de rejeu et un réseau cible pour stabiliser.
"""
import numpy as np

from . import game
from .equity import equity_vs_random
from .features import N_FEATURES, extract_features
from .network import QNetwork

REWARD_SCALE = 100 * game.BB  # les gains sont normalisés en centaines de BB


class ReplayBuffer:
    def __init__(self, capacity, feat_dim=N_FEATURES, n_actions=game.N_ACTIONS):
        self.capacity = capacity
        self.states = np.zeros((capacity, feat_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, feat_dim), dtype=np.float32)
        self.next_masks = np.ones((capacity, n_actions), dtype=bool)
        self.dones = np.zeros(capacity, dtype=bool)
        self.size = 0
        self.pos = 0

    def add(self, s, a, r, s2, mask2, done):
        i = self.pos
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = s2
        self.next_masks[i] = mask2
        self.dones[i] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.size, size=batch)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.next_masks[idx], self.dones[idx])


class DQNAgent:
    def __init__(self, seed=0, lr=3e-4, gamma=1.0, hidden=128):
        self.q = QNetwork(N_FEATURES, hidden, game.N_ACTIONS, seed=seed, lr=lr)
        self.target = QNetwork(N_FEATURES, hidden, game.N_ACTIONS, seed=seed + 1, lr=lr)
        self.target.set_weights(self.q.get_weights())
        self.gamma = gamma
        self.rng = np.random.default_rng(seed + 2)

    def act(self, features, legal_mask, eps=0.0):
        if eps > 0 and self.rng.random() < eps:
            return int(self.rng.choice(np.flatnonzero(legal_mask)))
        q = self.q.predict(features)[0]
        q = np.where(legal_mask, q, -1e9)
        return int(np.argmax(q))

    def learn(self, buffer, batch=128):
        s, a, r, s2, m2, done = buffer.sample(batch, self.rng)
        # Double DQN : le réseau en ligne CHOISIT l'action, le réseau cible
        # l'ÉVALUE. Sans ce découplage, le max sur des Q bruités surestime
        # systématiquement les actions à haute variance (all-in, calls légers).
        q_online = np.where(m2, self.q.predict(s2), -1e9)
        a_star = q_online.argmax(axis=1)
        q_target = self.target.predict(s2)
        bootstrap = q_target[np.arange(len(a_star)), a_star]
        targets = r + self.gamma * bootstrap * (~done)
        return self.q.train_step(s, a, targets)

    def update_target(self):
        self.target.set_weights(self.q.get_weights())


# --------------------------------------------------------------------- bots

class RandomBot:
    """Joue une action légale au hasard. Référence la plus faible."""

    def __init__(self, rng):
        self.rng = rng

    def act(self, hand, player):
        return int(self.rng.choice(hand.legal_actions()))


class CallBot:
    """Suit toujours (calling station). Punit les bluffs, jamais l'initiative."""

    def act(self, hand, player):
        return game.CHECK_CALL


class RuleBot:
    """Bot à règles basé sur l'équité et la cote du pot. Adversaire correct."""

    def __init__(self, rng, n_sims=100):
        self.rng = rng
        self.n_sims = n_sims

    def act(self, hand, player):
        legal = hand.legal_actions()
        equity = equity_vs_random(hand.hole[player], hand.board, self.n_sims, self.rng)
        to_call = hand.to_call(player)
        if to_call == 0:
            if game.RAISE_POT in legal and equity > 0.72 and self.rng.random() < 0.7:
                return game.RAISE_POT
            if game.RAISE_HALF in legal and equity > 0.55 and self.rng.random() < 0.6:
                return game.RAISE_HALF
            return game.CHECK_CALL
        pot_odds = to_call / (hand.pot + to_call)
        if game.ALL_IN in legal and equity > 0.85 and self.rng.random() < 0.5:
            return game.ALL_IN
        if game.RAISE_HALF in legal and equity > pot_odds + 0.25:
            return game.RAISE_HALF
        if equity > pot_odds:
            return game.CHECK_CALL
        return game.FOLD


class ManiacBot:
    """Hyper-agressif : mise et relance sans arrêt, bluffe, ne lâche presque
    jamais. Force l'IA à apprendre à payer (call down) face à l'agression —
    la première faille qu'un humain chercherait à exploiter."""

    def __init__(self, rng, n_sims=80):
        self.rng = rng
        self.n_sims = n_sims

    def act(self, hand, player):
        legal = hand.legal_actions()
        equity = equity_vs_random(hand.hole[player], hand.board, self.n_sims, self.rng)
        to_call = hand.to_call(player)
        pot_odds = to_call / (hand.pot + to_call) if to_call else 0.0
        r = self.rng.random()
        if game.ALL_IN in legal and (r < 0.04 or (equity > 0.62 and r < 0.25)):
            return game.ALL_IN
        if game.RAISE_POT in legal and r < 0.30:
            return game.RAISE_POT
        if game.RAISE_HALF in legal and r < 0.55:
            return game.RAISE_HALF
        if to_call and equity < pot_odds - 0.20 and r < 0.5:
            return game.FOLD
        return game.CHECK_CALL


class NetworkPolicy:
    """Politique basée sur un réseau (adversaire de self-play ou IA finale).

    `temperature` > 0 active une stratégie mixte : les actions dont les valeurs
    Q sont proches sont échantillonnées (softmax) au lieu de toujours jouer le
    maximum — l'IA devient imprévisible sur les décisions serrées sans
    sacrifier les décisions claires. Les Q étant en centaines de BB,
    temperature=0.004 mélange les actions à ~0,5 BB d'écart."""

    def __init__(self, weights_or_net, rng, eps=0.0, n_sims=100, temperature=0.0):
        if isinstance(weights_or_net, QNetwork):
            self.net = weights_or_net
        else:
            self.net = QNetwork.from_weights(weights_or_net)
        self.rng = rng
        self.eps = eps
        self.n_sims = n_sims
        self.temperature = temperature

    def act(self, hand, player):
        # Le réseau peut avoir moins de sorties que le jeu n'a d'actions (DQN
        # figé à 5 sorties face à un moteur à 7 actions) : il ne considère alors
        # que SES actions (indices 0..out_dim-1), les tailles ajoutées lui sont
        # invisibles — comportement voulu, il ne les a jamais apprises.
        x = extract_features(hand, player, self.rng, self.n_sims)
        q_all = self.net.predict(x)[0]
        na = len(q_all)
        mask = hand.legal_mask()[:na]
        if self.eps > 0 and self.rng.random() < self.eps:
            return int(self.rng.choice(np.flatnonzero(mask)))
        q = np.where(mask, q_all, -np.inf)
        if self.temperature > 0:
            z = (q - q.max()) / self.temperature
            probs = np.exp(z)
            probs /= probs.sum()
            return int(self.rng.choice(na, p=probs))
        return int(np.argmax(q))
