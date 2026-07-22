"""MCCFR parallèle : plusieurs ouvriers explorent, un maître fusionne.

Chaque cycle :
1. K processus ouvriers partent du blueprint courant et calculent chacun un
   bloc d'itérations dans leur coin (graines différentes → donnes différentes) ;
2. le maître fusionne leurs comptabilités : regrets et stratégies s'additionnent
   (fusion = somme des tables − (K−1) × base, régrets replanchers à 0) ;
3. évaluation + checkpoint, et on repart.

Le blueprint n'est remplacé qu'après une fusion réussie : un crash ne coûte
au pire qu'un cycle. Reprise : relancer la même commande.

Usage :
    python train_cfr_parallel.py --iters 1500000 --workers 4
"""
import argparse
import csv
import os
import pickle
import subprocess
import sys
import time

import numpy as np

from poker_ai.cfr import CFRPolicy, NodeStore, run_iteration
from train_cfr import play_match, safe_append
from poker_ai.agent import NetworkPolicy, RuleBot
from poker_ai.network import QNetwork

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- ouvrier

def run_worker(base_path, out_path, chunk, seed, n_sims):
    rng = np.random.default_rng(seed)
    if os.path.exists(base_path):
        store = NodeStore.load(base_path)
    else:
        store = NodeStore()
    target = store.iterations + chunk
    while store.iterations < target:
        run_iteration(store, rng, n_sims=n_sims)
    store.save(out_path)


# ----------------------------------------------------------------- fusion

def _load_raw(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def merge_workers(base_path, worker_paths, chunk):
    """Fusionne : final = Σ ouvriers − (K−1) × base. Renvoie un NodeStore."""
    k = len(worker_paths)
    if os.path.exists(base_path):
        base = _load_raw(base_path)
        base_nodes = base["nodes"]
        base_iters = base["iterations"]
    else:
        base_nodes, base_iters = {}, 0

    scale = -(k - 1)
    acc = {key: [r * scale, s * scale] for key, (r, s) in base_nodes.items()}
    del base_nodes

    for path in worker_paths:
        data = _load_raw(path)
        for key, (r, s) in data["nodes"].items():
            node = acc.get(key)
            if node is None:
                acc[key] = [r.copy(), s.copy()]
            else:
                node[0] += r
                node[1] += s
        del data

    store = NodeStore()
    store.iterations = base_iters + k * chunk
    empty_keys = []
    for key, node in acc.items():
        np.maximum(node[0], 0.0, out=node[0])  # regret matching+ après fusion
        np.maximum(node[1], 0.0, out=node[1])
        # Élagage : un nœud tout à zéro est identique à un nœud absent
        # (stratégie uniforme dans les deux cas) — inutile de le stocker.
        if not node[0].any() and not node[1].any():
            empty_keys.append(key)
    for key in empty_keys:
        del acc[key]
    store.nodes = acc
    return store


# ----------------------------------------------------------------- maître

def main():
    parser = argparse.ArgumentParser(description="MCCFR parallèle")
    parser.add_argument("--iters", type=int, default=1_500_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=25_000,
                        help="itérations par ouvrier et par cycle")
    parser.add_argument("--sims", type=int, default=160)
    parser.add_argument("--eval-hands", type=int, default=600)
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "models"))
    # mode interne : processus ouvrier
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--base", help=argparse.SUPPRESS)
    parser.add_argument("--dest", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.base, args.dest, args.chunk, args.seed, args.sims)
        return

    blueprint_path = os.path.join(args.out, "cfr_blueprint.pkl")
    progress_path = os.path.join(args.out, "cfr_progress.csv")
    metrics_path = os.path.join(args.out, "cfr_metrics.csv")
    work_dir = os.path.join(args.out, "cfr_workers")
    os.makedirs(work_dir, exist_ok=True)
    for path, header in [
        (progress_path, ["iterations", "situations", "iters_par_s", "eta_min"]),
        (metrics_path, ["iterations", "bb100_vs_regles", "err_regles",
                        "bb100_vs_dqn", "err_dqn"]),
    ]:
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    dqn_path = os.path.join(args.out, "model_v3_dqn.npz")
    dqn_net = QNetwork.load(dqn_path) if os.path.exists(dqn_path) else None

    iterations = 0
    if os.path.exists(blueprint_path):
        with open(blueprint_path, "rb") as f:
            iterations = pickle.load(f)["iterations"]
    print(f"MCCFR parallèle : {args.workers} ouvriers × {args.chunk} itérations/cycle, "
          f"départ à {iterations}, objectif {args.iters}", flush=True)

    t0 = time.time()
    start_iters = iterations
    while iterations < args.iters:
        cycle_start = time.time()

        # 1. les ouvriers travaillent en parallèle
        procs = []
        worker_paths = []
        for w in range(args.workers):
            dest = os.path.join(work_dir, f"worker_{w}.pkl")
            worker_paths.append(dest)
            cmd = [sys.executable, os.path.abspath(__file__), "--worker",
                   "--base", blueprint_path, "--dest", dest,
                   "--chunk", str(args.chunk), "--sims", str(args.sims),
                   "--seed", str(iterations * 100 + w)]
            procs.append(subprocess.Popen(cmd, cwd=BASE_DIR))
        failed = sum(p.wait() != 0 for p in procs)
        if failed:
            print(f"{failed} ouvrier(s) en échec — cycle rejoué", flush=True)
            continue

        # 2. fusion
        store = merge_workers(blueprint_path, worker_paths, args.chunk)
        iterations = store.iterations

        # 3. évaluation + checkpoint
        eval_rng = np.random.default_rng(10_000 + iterations)
        policy = CFRPolicy(store, eval_rng, n_sims=args.sims)
        bb_rule, se_rule = play_match(policy, RuleBot(eval_rng, 128),
                                      args.eval_hands, eval_rng)
        if dqn_net is not None:
            dqn = NetworkPolicy(dqn_net, eval_rng, eps=0.0, n_sims=128)
            bb_dqn, se_dqn = play_match(policy, dqn, args.eval_hands, eval_rng)
        else:
            bb_dqn, se_dqn = float("nan"), float("nan")

        store.save(blueprint_path)
        speed = (iterations - start_iters) / max(time.time() - t0, 1e-9)
        eta_min = (args.iters - iterations) / max(speed, 1e-9) / 60
        cycle_s = time.time() - cycle_start
        print(f"[{iterations:>8}/{args.iters}] {len(store.nodes):>7} situations "
              f"| bb/100 vs règles {bb_rule:+7.1f} (±{se_rule:.0f}) "
              f"vs DQN {bb_dqn:+7.1f} | {speed:.0f} it/s (cycle {cycle_s:.0f}s)",
              flush=True)
        safe_append(progress_path, [iterations, len(store.nodes),
                                    round(speed, 1), round(eta_min)])
        safe_append(metrics_path, [iterations, round(bb_rule, 1), round(se_rule, 1),
                                   round(bb_dqn, 1), round(se_dqn, 1)])
        del store, policy

    print("Objectif atteint.", flush=True)


if __name__ == "__main__":
    main()
