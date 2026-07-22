"""Purge la stratégie moyenne d'un blueprint en conservant les régrets.

La stratégie jouée par CFR est la MOYENNE de toutes ses stratégies successives ;
au début de l'entraînement, cette moyenne est polluée par des itérations quasi
aléatoires. Remettre la somme de stratégie à zéro (sans toucher aux régrets, où
vit le savoir) puis continuer l'entraînement reconstruit une moyenne fondée
uniquement sur du jeu mûr. C'est le geste qui avait fait décoller la v1.

Usage :
    python purge_average.py models/cfr_v2/cfr_blueprint.pkl
"""
import argparse
import sys

import numpy as np

from poker_ai.cfr import NodeStore, N_A


def main():
    parser = argparse.ArgumentParser(description="Purge de la moyenne d'un blueprint")
    parser.add_argument("path")
    args = parser.parse_args()

    store = NodeStore.load(args.path)
    before = len(store.nodes)
    empty = []
    for key, node in store.nodes.items():
        node[N_A:] = 0.0  # somme de stratégie remise à zéro (régrets intacts)
        if not node.any():  # nœud devenu entièrement nul -> élaguer
            empty.append(key)
    for key in empty:
        del store.nodes[key]
    store.save(args.path)
    print(f"Moyenne purgée : {before} situations -> {len(store.nodes)} "
          f"(élagué {len(empty)} nœuds vides), régrets conservés, "
          f"{store.iterations} itérations", flush=True)


if __name__ == "__main__":
    main()
