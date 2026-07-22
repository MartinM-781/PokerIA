"""Moteur de Texas Hold'em no-limit en tête-à-tête (heads-up).

Une partie = une main. Les deux joueurs repartent à 100 BB (200 jetons) à
chaque main. Le bouton poste la petite blind et parle en premier préflop,
en dernier après le flop.

Actions discrètes : FOLD, CHECK_CALL, RAISE_HALF (relance ½ pot),
RAISE_POT (relance pot), ALL_IN, plus une petite taille RAISE_THIRD (⅓ pot)
ajoutée à l'indice 5 — les indices 0..4 restent inchangés pour rester
compatibles avec le réseau DQN figé. Le ⅓ pot cible la « petite mise » que les
coachs ont identifiée comme la seule fuite exploitable. (Une seule taille
ajoutée : deux feraient exploser l'arbre en tabulaire — cf. RAISE_CAP.)
"""
import numpy as np

from .evaluator import evaluate_hand

FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT, ALL_IN, RAISE_THIRD = range(6)
N_ACTIONS = 6
ACTION_NAMES = ["se couche", "check/call", "relance ½ pot", "relance pot",
                "all-in", "relance ⅓ pot"]

# Fraction du pot (après call) misée par chaque action de relance, en
# arithmétique ENTIÈRE identique côté Rust. all-in traité à part.
RAISE_SIZES = (RAISE_THIRD, RAISE_HALF, RAISE_POT, ALL_IN)

# Plafond de relances DIMENSIONNÉES par tour d'enchères : au-delà, seul le tapis
# reste offert. Borne l'arbre (guerres de relances) comme les solveurs pros.
RAISE_CAP = 4

PREFLOP, FLOP, TURN, RIVER = range(4)
STREET_NAMES = ["préflop", "flop", "turn", "river"]

SB, BB = 1, 2
START_STACK = 200  # 100 BB


class HeadsUpHand:
    """Une main de heads-up. Appeler step(action) jusqu'à `terminal`."""

    def __init__(self, rng, button=0):
        deck = np.arange(52)
        rng.shuffle(deck)
        self.hole = [deck[0:2].tolist(), deck[2:4].tolist()]
        self.full_board = deck[4:9].tolist()
        self.button = button
        self.stacks = [START_STACK, START_STACK]
        self.invested = [0, 0]
        self.bets = [0, 0]           # mises du tour d'enchères en cours
        self.street = PREFLOP
        self.acted = [False, False]
        self.last_raise = BB
        self.raises_this_street = 0
        self.raise_counts = [0, 0]         # relances par joueur sur toute la main
        self.street_raise_counts = [0, 0]  # relances par joueur sur ce tour d'enchères
        self.last_aggressor = None         # dernier joueur à avoir misé/relancé
        self.terminal = False
        self.showdown = False
        self.winner = None           # 0/1, -1 si partage
        self.payoffs = None          # gain net en jetons pour chaque joueur
        self.scores = None           # scores à l'abattage
        self.history = []            # (street, joueur, action effective)
        self._commit(button, SB)
        self._commit(1 - button, BB)
        self.to_act = button

    # ------------------------------------------------------------------ état

    @property
    def board(self):
        return self.full_board[:[0, 3, 4, 5][self.street]]

    @property
    def pot(self):
        return self.invested[0] + self.invested[1]

    def to_call(self, p=None):
        p = self.to_act if p is None else p
        return min(max(self.bets[1 - p] - self.bets[p], 0), self.stacks[p])

    def legal_actions(self):
        p = self.to_act
        o = 1 - p
        legal = [CHECK_CALL]
        if self.bets[o] > self.bets[p]:
            legal.insert(0, FOLD)
        if self.stacks[p] > self.bets[o] - self.bets[p] and self.stacks[o] > 0:
            if self.raises_this_street < RAISE_CAP:
                # Le ⅓ pot n'est offert qu'en OUVERTURE du tour d'enchères :
                # comme taille de sur-relance, sa lente escalade du pot rallonge
                # les guerres de relances et fait exploser l'arbre (mesuré ×20).
                if self.raises_this_street == 0:
                    legal.append(RAISE_THIRD)
                legal += [RAISE_HALF, RAISE_POT, ALL_IN]
            else:  # plafond atteint : seul le tapis reste (borne l'arbre)
                legal.append(ALL_IN)
        return legal

    def legal_mask(self):
        m = np.zeros(N_ACTIONS, dtype=bool)
        m[self.legal_actions()] = True
        return m

    # ---------------------------------------------------------------- action

    def step(self, action):
        assert not self.terminal, "la main est terminée"
        p = self.to_act
        o = 1 - p
        to_call = self.bets[o] - self.bets[p]

        if action == FOLD and to_call > 0:
            self.history.append((self.street, p, FOLD))
            self._finish_fold(winner=o)
            return

        can_raise = self.stacks[p] > to_call and self.stacks[o] > 0
        if action in RAISE_SIZES and not can_raise:
            action = CHECK_CALL  # garde-fou : action illégale rétrogradée

        if action in (FOLD, CHECK_CALL):  # FOLD sans mise à payer = check
            self.history.append((self.street, p, CHECK_CALL))
            self._commit(p, to_call)
            self.acted[p] = True
        else:
            self.history.append((self.street, p, action))
            pot_after_call = self.pot + to_call
            if action == RAISE_THIRD:
                raise_by = pot_after_call // 3
            elif action == RAISE_HALF:
                raise_by = pot_after_call // 2
            elif action == RAISE_POT:
                raise_by = pot_after_call
            else:
                raise_by = self.stacks[p]
            raise_by = max(raise_by, self.last_raise, BB)
            self._commit(p, min(to_call + raise_by, self.stacks[p]))
            actual_raise = self.bets[p] - self.bets[o]
            if actual_raise > 0:
                self.last_raise = max(actual_raise, BB)
                self.raises_this_street += 1
                self.raise_counts[p] += 1
                self.street_raise_counts[p] += 1
                self.last_aggressor = p
            self.acted = [False, False]
            self.acted[p] = True

        low = 0 if self.bets[0] <= self.bets[1] else 1
        settled = self.bets[0] == self.bets[1] or self.stacks[low] == 0
        if self.acted[0] and self.acted[1] and settled:
            self._next_street()
        else:
            self.to_act = o

    # -------------------------------------------------------------- interne

    def _commit(self, p, amount):
        amount = min(amount, self.stacks[p])
        self.stacks[p] -= amount
        self.invested[p] += amount
        self.bets[p] += amount

    def _next_street(self):
        if self.street == RIVER or self.stacks[0] == 0 or self.stacks[1] == 0:
            self.street = RIVER  # tapis : on déroule le board jusqu'au bout
            self._showdown()
            return
        self.street += 1
        self.bets = [0, 0]
        self.acted = [False, False]
        self.last_raise = 0
        self.raises_this_street = 0
        self.street_raise_counts = [0, 0]
        self.to_act = 1 - self.button

    def _showdown(self):
        # Rembourse une éventuelle mise non suivie (all-in pour moins)
        diff = self.invested[0] - self.invested[1]
        if diff > 0:
            self.stacks[0] += diff
            self.invested[0] -= diff
        elif diff < 0:
            self.stacks[1] -= diff
            self.invested[1] += diff
        self.scores = [evaluate_hand(self.hole[i] + self.full_board) for i in (0, 1)]
        pot = self.pot
        if self.scores[0] > self.scores[1]:
            self.winner = 0
            self.stacks[0] += pot
        elif self.scores[1] > self.scores[0]:
            self.winner = 1
            self.stacks[1] += pot
        else:
            self.winner = -1
            self.stacks[0] += pot // 2
            self.stacks[1] += pot - pot // 2
        self.showdown = True
        self._finish()

    def _finish_fold(self, winner):
        # Rembourse la mise non suivie du gagnant (uncalled bet) avant de lui
        # attribuer le pot, comme au showdown : le pot ne contient que les jetons
        # réellement engagés par les deux joueurs.
        diff = self.invested[winner] - self.invested[1 - winner]
        if diff > 0:
            self.stacks[winner] += diff
            self.invested[winner] -= diff
        self.winner = winner
        self.stacks[winner] += self.pot
        self._finish()

    def _finish(self):
        self.terminal = True
        self.payoffs = [self.stacks[i] - START_STACK for i in (0, 1)]

    def clone(self):
        """Copie légère et indépendante — pour l'exploration d'arbre (CFR).
        `full_board` est partagé (jamais modifié après la donne)."""
        h = object.__new__(HeadsUpHand)
        h.hole = [self.hole[0][:], self.hole[1][:]]
        h.full_board = self.full_board
        h.button = self.button
        h.stacks = self.stacks[:]
        h.invested = self.invested[:]
        h.bets = self.bets[:]
        h.street = self.street
        h.acted = self.acted[:]
        h.last_raise = self.last_raise
        h.raises_this_street = self.raises_this_street
        h.raise_counts = self.raise_counts[:]
        h.street_raise_counts = self.street_raise_counts[:]
        h.last_aggressor = self.last_aggressor
        h.terminal = self.terminal
        h.showdown = self.showdown
        h.winner = self.winner
        h.payoffs = None if self.payoffs is None else self.payoffs[:]
        h.scores = None if self.scores is None else self.scores[:]
        h.history = self.history[:]
        h.to_act = self.to_act
        return h
