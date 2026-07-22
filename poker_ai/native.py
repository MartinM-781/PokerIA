"""Pont vers le cœur compilé (poker_native, Rust) avec repli Python pur.

Le module natif porte le chemin chaud du MCCFR : moteur de jeu, évaluateur,
équité Monte-Carlo, buckets v2 et traversée — parité validée (scores et clés
identiques, moteur au jeton près). Speedup mesuré : ~×90.
"""
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
    ns.iterations = store.iterations
    ns.run(chunk, seed, n_sims)
    keys, values = ns.export_nodes()
    store.nodes = {k: values[i] for i, k in enumerate(keys)}
    store.iterations = ns.iterations
    return store
