"""Boucle d'entraînement par self-play.

L'agent apprend en jouant contre un mélange d'adversaires :
- des copies gelées de lui-même (self-play, la source principale de progrès),
- un bot à règles, un bot suiveur et un bot aléatoire (pour l'ancrer contre
  des styles variés et éviter les stratégies dégénérées).
"""
import csv
import os
import time
from collections import deque

import numpy as np

from . import game
from .agent import (REWARD_SCALE, CallBot, DQNAgent, ManiacBot, NetworkPolicy,
                    RandomBot, ReplayBuffer, RuleBot)
from .features import extract_features
from .game import HeadsUpHand


def _safe_append(path, row):
    """Ajoute une ligne CSV en tolérant un verrou temporaire (Excel, antivirus…) :
    un échec d'écriture de suivi ne doit jamais interrompre l'entraînement."""
    for _ in range(3):
        try:
            with open(path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            return True
        except OSError:
            time.sleep(0.2)
    return False


def _safe_save(net, path):
    """Sauvegarde atomique et tolérante du modèle (tmp + replace, avec reprises)."""
    tmp = path + ".tmp.npz"
    for _ in range(3):
        try:
            net.save(tmp)
            os.replace(tmp, path)
            return True
        except OSError:
            time.sleep(0.5)
    return False


def play_hand(policy_fn, opponent, rng, button, n_sims, collect=None):
    """Joue une main : le joueur 0 suit `policy_fn(features, mask)`, le joueur 1
    est un bot. Si `collect` est une liste, y ajoute les paires (features, mask,
    action) du joueur 0. Renvoie le gain du joueur 0 en jetons."""
    hand = HeadsUpHand(rng, button=button)
    while not hand.terminal:
        if hand.to_act == 0:
            x = extract_features(hand, 0, rng, n_sims)
            mask = hand.legal_mask()
            a = policy_fn(x, mask)
            if collect is not None:
                collect.append((x, mask, a))
            hand.step(a)
        else:
            hand.step(opponent.act(hand, 1))
    return hand.payoffs[0]


def store_hand(buffer, decisions, payoff):
    """Transforme les décisions d'une main en transitions DQN."""
    reward = payoff / REWARD_SCALE
    for i, (x, _mask, a) in enumerate(decisions):
        if i + 1 < len(decisions):
            nx, nmask, _ = decisions[i + 1]
            buffer.add(x, a, 0.0, nx, nmask, False)
        else:
            buffer.add(x, a, reward, np.zeros_like(x), np.ones(game.N_ACTIONS, bool), True)


def evaluate_policy(agent, opponent, n_hands, rng, n_sims=100):
    """bb/100 de l'agent (glouton) contre un bot, et erreur-type."""
    results = np.zeros(n_hands)
    greedy = lambda x, m: agent.act(x, m, eps=0.0)
    for i in range(n_hands):
        results[i] = play_hand(greedy, opponent, rng, button=i % 2, n_sims=n_sims)
    bb = results / game.BB
    bb100 = bb.mean() * 100
    stderr = bb.std() / np.sqrt(n_hands) * 100
    return bb100, stderr


def train(n_hands=40000, seed=0, lr=3e-4, hidden=128, n_sims=100,
          out_dir="models", eval_every=5000, eval_hands=800,
          progress_every=100, resume_path=None, log=None):
    if log is None:
        log = lambda m: print(m, flush=True)
    rng = np.random.default_rng(seed)
    agent = DQNAgent(seed=seed, lr=lr, hidden=hidden)
    buffer = ReplayBuffer(200_000)
    pool = deque(maxlen=10)  # copies gelées pour le self-play

    eps_start, eps_end = 1.0, 0.05
    if resume_path and os.path.exists(resume_path):
        from .network import QNetwork
        weights = QNetwork.load(resume_path).get_weights()
        agent.q.set_weights(weights)
        agent.target.set_weights(weights)
        pool.append(weights)
        eps_start = 0.30  # on repart d'un modèle déjà compétent : moins d'exploration
        log(f"Reprise depuis {resume_path} (exploration réduite à {eps_start:.0%})")

    rule_bot = RuleBot(rng, n_sims=n_sims)
    call_bot = CallBot()
    random_bot = RandomBot(rng)
    maniac_bot = ManiacBot(rng)

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["mains", "epsilon", "perte", "bb100_vs_regles", "bb100_vs_call", "bb100_vs_aleatoire"])

    # Ticker léger : une ligne toutes les `progress_every` mains, coût nul.
    # bb100_recent est mesuré sur les mains d'ENTRAÎNEMENT (exploration comprise) :
    # c'est un indicateur d'ambiance, la vraie force est dans metrics.csv.
    progress_path = os.path.join(out_dir, "progress.csv")
    with open(progress_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["mains", "epsilon", "perte", "bb100_recent", "mains_par_s", "eta_min"])
    recent_payoffs = deque(maxlen=2000)

    eps_decay_hands = int(n_hands * 0.6)
    losses = deque(maxlen=500)
    t0 = time.time()

    for i in range(1, n_hands + 1):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * i / max(eps_decay_hands, 1))

        r = rng.random()
        if pool and r < 0.52:
            opponent = NetworkPolicy(pool[int(rng.integers(len(pool)))], rng,
                                     eps=0.05, n_sims=64)
        elif r < 0.76:
            opponent = rule_bot     # le seul prof qui punit le « je ne me couche jamais »
        elif r < 0.84:
            opponent = maniac_bot
        elif r < 0.92:
            opponent = call_bot
        else:
            opponent = random_bot

        decisions = []
        payoff = play_hand(lambda x, m: agent.act(x, m, eps),
                           opponent, rng, button=i % 2, n_sims=n_sims,
                           collect=decisions)
        store_hand(buffer, decisions, payoff)
        recent_payoffs.append(payoff / game.BB)

        if i % progress_every == 0:
            speed = i / max(time.time() - t0, 1e-9)
            loss = float(np.mean(losses)) if losses else float("nan")
            _safe_append(progress_path, [
                i, round(eps, 3), round(loss, 5),
                round(float(np.mean(recent_payoffs)) * 100, 1),
                round(speed, 1), round((n_hands - i) / speed / 60),
            ])

        if buffer.size >= 2000:
            for _ in range(3):
                losses.append(agent.learn(buffer, batch=128))

        if i % 500 == 0:
            agent.update_target()
        if i % 3000 == 0:
            pool.append(agent.q.get_weights())
        # Décroissance du taux d'apprentissage : affine la stratégie en fin de course
        if i == int(n_hands * 0.6) or i == int(n_hands * 0.85):
            agent.q.lr *= 0.4

        if i % eval_every == 0 or i == n_hands:
            eval_rng = np.random.default_rng(seed + 10_000 + i)
            bb_rule, se_rule = evaluate_policy(agent, RuleBot(eval_rng, n_sims), eval_hands, eval_rng, n_sims)
            bb_call, _ = evaluate_policy(agent, call_bot, eval_hands, eval_rng, n_sims)
            bb_rand, _ = evaluate_policy(agent, RandomBot(eval_rng), eval_hands, eval_rng, n_sims)
            loss = float(np.mean(losses)) if losses else float("nan")
            speed = i / (time.time() - t0)
            log(f"[{i:>6}/{n_hands}] eps={eps:.2f} perte={loss:.4f} "
                f"| bb/100 vs règles {bb_rule:+7.1f} (±{se_rule:.0f}) "
                f"vs call {bb_call:+7.1f} vs aléatoire {bb_rand:+7.1f} "
                f"| {speed:.0f} mains/s")
            _safe_append(metrics_path, [i, round(eps, 3), round(loss, 5),
                                        round(bb_rule, 1), round(bb_call, 1), round(bb_rand, 1)])
            _safe_save(agent.q, os.path.join(out_dir, "model.npz"))

    _safe_save(agent.q, os.path.join(out_dir, "model.npz"))
    log(f"Modèle sauvegardé dans {os.path.join(out_dir, 'model.npz')}")
    return agent
