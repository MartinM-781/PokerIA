"""Calcule le blueprint MCCFR (stratégie proche de l'équilibre de Nash).

Usage :
    python train_cfr.py                     # 1 500 000 itérations
    python train_cfr.py --resume           # reprend models/cfr_blueprint.pkl
    python train_cfr.py --iters 3000000    # plus long = plus proche de l'équilibre

Suivi :
    models/cfr_progress.csv   (ticker toutes les 1 000 itérations)
    models/cfr_metrics.csv    (force mesurée à chaque checkpoint)
"""
import argparse
import csv
import os
import time

import numpy as np

from poker_ai import game
from poker_ai.agent import NetworkPolicy, RuleBot
from poker_ai.cfr import CFRPolicy, NodeStore, run_iteration
from poker_ai.game import HeadsUpHand
from poker_ai.network import QNetwork

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def safe_append(path, row):
    for _ in range(3):
        try:
            with open(path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            return
        except OSError:
            time.sleep(0.2)


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
    parser = argparse.ArgumentParser(description="Entraînement MCCFR")
    parser.add_argument("--iters", type=int, default=1_500_000)
    parser.add_argument("--sims", type=int, default=160,
                        help="simulations Monte-Carlo par bucket d'équité")
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-hands", type=int, default=600)
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "models"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="reprend le blueprint existant")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    blueprint_path = os.path.join(args.out, "cfr_blueprint.pkl")
    progress_path = os.path.join(args.out, "cfr_progress.csv")
    metrics_path = os.path.join(args.out, "cfr_metrics.csv")

    rng = np.random.default_rng(args.seed)
    if args.resume and os.path.exists(blueprint_path):
        store = NodeStore.load(blueprint_path)
        print(f"Reprise : {store.iterations} itérations, "
              f"{len(store.nodes)} situations connues", flush=True)
    else:
        store = NodeStore()
        for path, header in [
            (progress_path, ["iterations", "situations", "iters_par_s", "eta_min"]),
            (metrics_path, ["iterations", "bb100_vs_regles", "err_regles",
                            "bb100_vs_dqn", "err_dqn"]),
        ]:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    dqn_path = os.path.join(args.out, "model_v3_dqn.npz")
    dqn_net = QNetwork.load(dqn_path) if os.path.exists(dqn_path) else None

    start_iters = store.iterations
    t0 = time.time()
    print(f"MCCFR : objectif {args.iters} itérations "
          f"(Ctrl+C : le dernier checkpoint est conservé)", flush=True)

    try:
        while store.iterations < args.iters:
            run_iteration(store, rng, n_sims=args.sims)
            i = store.iterations

            if i % 1000 == 0:
                speed = (i - start_iters) / max(time.time() - t0, 1e-9)
                eta = (args.iters - i) / max(speed, 1e-9) / 60
                safe_append(progress_path,
                            [i, len(store.nodes), round(speed, 1), round(eta)])

            if i % args.eval_every == 0 or i == args.iters:
                eval_rng = np.random.default_rng(10_000 + i)
                policy = CFRPolicy(store, eval_rng, n_sims=args.sims)
                bb_rule, se_rule = play_match(
                    policy, RuleBot(eval_rng, 128), args.eval_hands, eval_rng)
                if dqn_net is not None:
                    dqn = NetworkPolicy(dqn_net, eval_rng, eps=0.0, n_sims=128)
                    bb_dqn, se_dqn = play_match(policy, dqn, args.eval_hands, eval_rng)
                else:
                    bb_dqn, se_dqn = float("nan"), float("nan")
                speed = (i - start_iters) / max(time.time() - t0, 1e-9)
                print(f"[{i:>8}/{args.iters}] {len(store.nodes):>7} situations "
                      f"| bb/100 vs règles {bb_rule:+7.1f} (±{se_rule:.0f}) "
                      f"vs DQN {bb_dqn:+7.1f} | {speed:.0f} it/s", flush=True)
                safe_append(metrics_path, [i, round(bb_rule, 1), round(se_rule, 1),
                                           round(bb_dqn, 1), round(se_dqn, 1)])
                store.save(blueprint_path)
    except KeyboardInterrupt:
        print("\nInterrompu — sauvegarde…", flush=True)

    store.save(blueprint_path)
    print(f"Blueprint sauvegardé : {blueprint_path} "
          f"({store.iterations} itérations, {len(store.nodes)} situations)", flush=True)


if __name__ == "__main__":
    main()
