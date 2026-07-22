"""Entraîne l'IA de poker par self-play.

Usage :
    python train.py                     # 40 000 mains (~10 min)
    python train.py --hands 100000      # entraînement plus long = IA plus forte
"""
import argparse

from poker_ai.training import train


def main():
    parser = argparse.ArgumentParser(description="Entraînement de l'IA de poker")
    parser.add_argument("--hands", type=int, default=40000, help="nombre de mains de self-play")
    parser.add_argument("--seed", type=int, default=0, help="graine aléatoire")
    parser.add_argument("--lr", type=float, default=3e-4, help="taux d'apprentissage")
    parser.add_argument("--hidden", type=int, default=128, help="neurones par couche cachée")
    parser.add_argument("--sims", type=int, default=100, help="simulations Monte-Carlo par décision")
    parser.add_argument("--out", default="models", help="dossier de sortie")
    parser.add_argument("--eval-every", type=int, default=5000, help="fréquence d'évaluation")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="fréquence du ticker de progression (models/progress.csv)")
    parser.add_argument("--resume", default=None, metavar="MODELE",
                        help="repart des poids d'un modèle existant (ex. models/model.npz)")
    args = parser.parse_args()

    print(f"Entraînement : {args.hands} mains de self-play (Ctrl+C pour arrêter, "
          f"le dernier point de contrôle est conservé)\n")
    try:
        train(n_hands=args.hands, seed=args.seed, lr=args.lr, hidden=args.hidden,
              n_sims=args.sims, out_dir=args.out, eval_every=args.eval_every,
              progress_every=args.progress_every, resume_path=args.resume)
    except KeyboardInterrupt:
        print("\nInterrompu — le dernier point de contrôle est dans "
              f"{args.out}/model.npz")


if __name__ == "__main__":
    main()
