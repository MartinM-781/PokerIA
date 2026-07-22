"""Entraînement MCCFR natif MONO-PROCESSUS : les ouvriers sont des threads
dans le moteur Rust, fusionnés en RAM. Aucun fichier ouvrier n'est écrit sur
le disque — seul le blueprint est sauvegardé, une fois par point de contrôle.

C'est la voie recommandée quand `poker_native` est compilé : plus rapide (pas
de sérialisation Python↔Rust par cycle) et bien plus douce pour le disque que
`train_cfr_parallel.py` (qui écrit une copie du blueprint par ouvrier à chaque
cycle).

Usage :
    python train_cfr_native.py --iters 12000000 --workers 6 --chunk 250000
"""
import argparse
import os
import time

import numpy as np

from poker_ai import game
from poker_ai.agent import NetworkPolicy, RuleBot
from poker_ai.cfr import CFRPolicy
from poker_ai.game import HeadsUpHand
from poker_ai.native import NativeTrainer
from poker_ai.network import QNetwork
from train_cfr import safe_append

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def play_match(policy, opponent, n_hands, rng):
    results = np.zeros(n_hands)
    for i in range(n_hands):
        hand = HeadsUpHand(rng, button=i % 2)
        while not hand.terminal:
            actor = policy if hand.to_act == 0 else opponent
            hand.step(actor.act(hand, hand.to_act))
        results[i] = hand.payoffs[0] / game.BB
    return results.mean() * 100, results.std() / np.sqrt(n_hands) * 100


def main():
    parser = argparse.ArgumentParser(description="MCCFR natif mono-processus")
    parser.add_argument("--iters", type=int, default=12_000_000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=250_000,
                        help="itérations par thread et par cycle")
    parser.add_argument("--sims", type=int, default=160)
    parser.add_argument("--eval-hands", type=int, default=600)
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "models", "cfr_v2"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    blueprint_path = os.path.join(args.out, "cfr_blueprint.pkl")
    progress_path = os.path.join(args.out, "cfr_progress.csv")
    metrics_path = os.path.join(args.out, "cfr_metrics.csv")
    import csv
    for path, header in [
        (progress_path, ["iterations", "situations", "iters_par_s", "eta_min"]),
        (metrics_path, ["iterations", "bb100_vs_regles", "err_regles",
                        "bb100_vs_dqn", "err_dqn"]),
    ]:
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    dqn_path = os.path.join(args.out, "model_v3_dqn.npz")
    if not os.path.exists(dqn_path):
        dqn_path = os.path.join(BASE_DIR, "models", "model_v3_dqn.npz")
    dqn_net = QNetwork.load(dqn_path) if os.path.exists(dqn_path) else None

    trainer = NativeTrainer(blueprint_path)
    start_iters = trainer.iterations
    print(f"MCCFR natif : {args.workers} threads × {args.chunk} itérations/cycle, "
          f"départ à {start_iters}, objectif {args.iters} "
          f"(mono-processus, aucun fichier ouvrier)", flush=True)

    per_cycle = args.workers * args.chunk
    t0 = time.time()
    while trainer.iterations < args.iters:
        seed = args.seed + trainer.iterations  # graine distincte par cycle
        trainer.run_cycle(args.workers, args.chunk, seed, args.sims)
        i = trainer.iterations
        speed = (i - start_iters) / max(time.time() - t0, 1e-9)
        eta = (args.iters - i) / max(speed, 1e-9) / 60
        safe_append(progress_path, [i, len(trainer), round(speed, 1), round(eta)])

        # Point de contrôle : snapshot → évaluation → sauvegarde
        snap = trainer.snapshot()
        eval_rng = np.random.default_rng(10_000 + i)
        policy = CFRPolicy(snap, eval_rng, n_sims=args.sims)
        bb_rule, se_rule = play_match(policy, RuleBot(eval_rng, 128), args.eval_hands, eval_rng)
        if dqn_net is not None:
            dqn = NetworkPolicy(dqn_net, eval_rng, eps=0.0, n_sims=128)
            bb_dqn, se_dqn = play_match(policy, dqn, args.eval_hands, eval_rng)
        else:
            bb_dqn, se_dqn = float("nan"), float("nan")
        safe_append(metrics_path, [i, round(bb_rule, 1), round(se_rule, 1),
                                   round(bb_dqn, 1), round(se_dqn, 1)])
        snap.save(blueprint_path)
        print(f"[{i:>9}/{args.iters}] {len(trainer):>7} situations "
              f"| bb/100 vs règles {bb_rule:+7.1f} (±{se_rule:.0f}) "
              f"vs DQN {bb_dqn:+7.1f} | {speed:.0f} it/s", flush=True)
        del snap, policy

    print("Objectif atteint.", flush=True)


if __name__ == "__main__":
    main()
