"""Cartes : un entier 0..51. rang = carte // 4 (0 = Deux ... 12 = As), couleur = carte % 4."""

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "♠♥♦♣"  # pique, coeur, carreau, trefle
RED_SUITS = (1, 2)


def card_str(card, color=False):
    """Représentation lisible d'une carte, ex. 'A♠'."""
    s = RANK_CHARS[card // 4] + SUIT_CHARS[card % 4]
    if color and card % 4 in RED_SUITS:
        return f"\033[91m{s}\033[0m"
    return s


def cards_str(cards, color=False):
    return " ".join(card_str(c, color) for c in cards)
