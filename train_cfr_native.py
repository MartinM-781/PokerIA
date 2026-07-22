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
import threading
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

RAM_CEILING = 0.93  # ne jamais utiliser plus de 93 % de la RAM machine


def _ram_gb():
    """(disponible, totale) en Go — via l'API Windows, sans dépendance."""
    import ctypes

    class MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64)]

    m = MemStatus()
    m.dwLength = ctypes.sizeof(MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1e9, m.ullTotalPhys / 1e9


def adaptive_workers(want, n_nodes, growth_nodes_per_thread):
    """Nombre de threads qui tient sous le plafond RAM : chaque thread clone la
    table (~140 o/nœud) puis la fait grossir pendant son bloc ; la fusion garde
    toutes les copies vivantes brièvement. On garde 10 % de la machine libre."""
    avail, total = _ram_gb()
    # `avail` (ullAvailPhys) inclut déjà le cache disque réclamable — c'est la
    # vraie mémoire mobilisable, pas le « libre strict ».
    budget = max(avail - (1.0 - RAM_CEILING) * total, 0.5)
    # ~130 o/nœud en RAM (clé String + 12 float32 + surcoût HashMap), mesuré.
    per_thread = (n_nodes + growth_nodes_per_thread) * 130 / 1e9 + 0.05
    # plancher 1 thread de marge (au lieu de -1) : moins un cran de prudence.
    return max(2, min(want, int(budget / per_thread)))


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
    parser.add_argument("--workers", type=int, default=12,
                        help="threads de calcul (pic vers 10-12 sur cette machine)")
    parser.add_argument("--chunk", type=int, default=200_000,
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
          f"(mono-processus ; éval superposée au calcul)", flush=True)

    def checkpoint(snap):
        """Évalue et sauvegarde un instantané. Tourne sur le thread principal
        PENDANT que les threads Rust du cycle suivant calculent (GIL relâché) :
        les cœurs ne s'arrêtent jamais pour l'évaluation."""
        i = snap.iterations
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
        speed = (i - start_iters) / max(time.time() - t0, 1e-9)
        eta = (args.iters - i) / max(speed, 1e-9) / 60
        safe_append(progress_path, [i, len(snap.nodes), round(speed, 1), round(eta)])
        print(f"[{i:>9}/{args.iters}] {len(snap.nodes):>7} situations "
              f"| bb/100 vs règles {bb_rule:+7.1f} (±{se_rule:.0f}) "
              f"vs DQN {bb_dqn:+7.1f} | {speed:.0f} it/s", flush=True)

    t0 = time.time()
    pending = None  # instantané en attente d'évaluation (superposé au calcul)
    growth_per_thread = args.chunk * 110  # estimation initiale, affinée par cycle
    while trainer.iterations < args.iters:
        seed = args.seed + trainer.iterations  # graine distincte par cycle
        n_before = len(trainer)
        workers = adaptive_workers(args.workers, n_before, growth_per_thread)
        if workers < args.workers:
            print(f"  (RAM : {workers} threads ce cycle au lieu de {args.workers})",
                  flush=True)
        # Lance le calcul du cycle dans un thread : run_parallel relâche le GIL,
        # donc le checkpoint du cycle précédent tourne en parallèle ci-dessous.
        compute = threading.Thread(
            target=trainer.run_cycle, args=(workers, args.chunk, seed, args.sims))
        compute.start()
        if pending is not None:
            checkpoint(pending)
            pending = None
        compute.join()
        growth_per_thread = max((len(trainer) - n_before) // max(workers, 1), 1000)
        pending = trainer.snapshot()

    if pending is not None:  # dernier checkpoint (plus rien à superposer)
        checkpoint(pending)
    print("Objectif atteint.", flush=True)


if __name__ == "__main__":
    main()
