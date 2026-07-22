"""Pont vers le cœur compilé (poker_native, Rust) avec repli Python pur.

Le module natif porte le chemin chaud du MCCFR : moteur de jeu, évaluateur,
équité Monte-Carlo, buckets v2 et traversée — parité validée (scores et clés
identiques, moteur au jeton près). Speedup mesuré : ~×90.
"""
import os

import numpy as np

try:
    import poker_native as _native
    NATIVE_AVAILABLE = True
except ImportError:  # pas compilé sur cette machine : repli Python pur
    _native = None
    NATIVE_AVAILABLE = False


def run_chunk_native(store, chunk, seed, n_sims):
    """Fait tourner `chunk` itérations MCCFR côté Rust sur le contenu de
    `store` (poker_ai.cfr.NodeStore), et rapatrie la table mise à jour."""
    if not NATIVE_AVAILABLE:
        raise RuntimeError("poker_native n'est pas installé (voir native/README)")
    ns = _native.NativeStore(store.version)
    if store.nodes:
        keys = list(store.nodes.keys())
        values = np.stack(list(store.nodes.values())).astype(np.float32, copy=False)
        ns.load_nodes(keys, values)
        store.nodes = {}  # libère la copie Python pendant que Rust calcule
        del keys, values
    ns.iterations = store.iterations
    ns.run(chunk, seed, n_sims)
    keys, values = ns.export_nodes()
    store.nodes = {k: values[i] for i, k in enumerate(keys)}
    store.iterations = ns.iterations
    return store


class NativeTrainer:
    """Table MCCFR vivant côté Rust en permanence : les cycles parallèles
    tournent en threads (aucun fichier ouvrier sur le disque), et on n'exporte
    la table vers Python que pour les points de contrôle."""

    def __init__(self, blueprint_path=None):
        if not NATIVE_AVAILABLE:
            raise RuntimeError("poker_native n'est pas installé (voir native/)")
        from .cfr import LATEST_VERSION, NodeStore
        if blueprint_path and os.path.exists(blueprint_path):
            store = NodeStore.load(blueprint_path)  # migre l'ancien format au besoin
            self.store = _native.NativeStore(store.version)
            if store.nodes:
                keys = list(store.nodes.keys())
                values = np.stack(list(store.nodes.values())).astype(np.float32, copy=False)
                self.store.load_nodes(keys, values)
            self.store.iterations = store.iterations
            del store, keys, values
        else:
            self.store = _native.NativeStore(LATEST_VERSION)

    @property
    def iterations(self):
        return self.store.iterations

    def __len__(self):
        return len(self.store)

    def run_cycle(self, workers, chunk, base_seed, n_sims):
        """Un cycle : `workers` threads × `chunk` itérations, fusionnés en RAM."""
        self.store.run_parallel(workers, chunk, base_seed, n_sims)

    def snapshot(self):
        """Copie instantanée de la table sous forme de NodeStore (pour évaluer
        et sauvegarder). N'interrompt pas l'entraînement."""
        from .cfr import NodeStore
        keys, values = self.store.export_nodes()
        snap = NodeStore(self.store.version)
        snap.iterations = self.store.iterations
        snap.nodes = {k: values[i] for i, k in enumerate(keys)}
        return snap
