"""Serveur local du dashboard : table de poker graphique + statistiques.

Usage :
    python server.py              # puis ouvrir http://localhost:8777
    python server.py --sims 800   # IA plus précise (un peu plus lente)
"""
import argparse
import csv
import json
import os
import pickle as pickle_module
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from poker_ai import game
from poker_ai.agent import NetworkPolicy
from poker_ai.evaluator import CATEGORY_NAMES, hand_category
from poker_ai.game import ACTION_NAMES, STREET_NAMES, HeadsUpHand
from poker_ai.network import QNetwork

HUMAN, AI = 0, 1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
SESSION_PATH = os.path.join(BASE_DIR, "models", "session.json")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.csv")

STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/training": ("training.html", "text/html; charset=utf-8"),
    "/training.js": ("training.js", "text/javascript; charset=utf-8"),
}
PROGRESS_PATH = os.path.join(BASE_DIR, "models", "progress.csv")
# Le suivi CFR pointe sur l'entraînement le plus récent (v2 si présent).
_CFR_DIR = (os.path.join(BASE_DIR, "models", "cfr_v2")
            if os.path.exists(os.path.join(BASE_DIR, "models", "cfr_v2", "cfr_progress.csv"))
            else os.path.join(BASE_DIR, "models"))
CFR_PROGRESS_PATH = os.path.join(_CFR_DIR, "cfr_progress.csv")
CFR_METRICS_PATH = os.path.join(_CFR_DIR, "cfr_metrics.csv")


class ApiError(Exception):
    pass


class GameSession:
    """Une session de jeu humain contre IA, protégée par un verrou."""

    def __init__(self, model_path, n_sims, seed=None, temperature=0.004):
        self.rng = np.random.default_rng(seed)
        self.model_path = model_path
        self.n_sims = n_sims
        self.temperature = temperature
        self.model_mtime = os.path.getmtime(model_path)
        self.ai = self._build_policy()
        self.lock = threading.Lock()
        self.hand = None
        self.hand_recorded = False
        self.button = AI  # inversé au début de chaque main → l'humain commence au bouton
        self.hand_id = 0
        self.history = []          # gain humain en BB, par main
        self.biggest_pot = 0
        self.showdowns = {"gagnes": 0, "perdus": 0, "partages": 0}
        self._load_session()

    def _build_policy(self):
        """Cerveau selon l'extension : .pkl = blueprint CFR, .npz = réseau DQN."""
        if self.model_path.endswith(".pkl"):
            from poker_ai.cfr import CFRPolicy
            return CFRPolicy(self.model_path, self.rng, n_sims=max(self.n_sims, 160))
        return NetworkPolicy(QNetwork.load(self.model_path), self.rng, eps=0.0,
                             n_sims=self.n_sims, temperature=self.temperature)

    def _maybe_reload_model(self):
        """Recharge le modèle si le fichier a changé (entraînement en cours) —
        on joue ainsi contre la version la plus récente de l'IA. Le chargement
        (long pour un blueprint CFR) se fait en arrière-plan : les mains en
        cours continuent sur l'ancienne version, la bascule est transparente."""
        try:
            mtime = os.path.getmtime(self.model_path)
        except OSError:
            return
        if mtime == self.model_mtime or getattr(self, "_reloading", False):
            return
        self._reloading = True

        def _load():
            try:
                policy = self._build_policy()
                self.ai = policy          # affectation atomique
                self.model_mtime = mtime
            except (OSError, ValueError, KeyError, EOFError,
                    pickle_module.UnpicklingError):
                pass  # fichier en cours d'écriture : on garde le modèle actuel
            finally:
                self._reloading = False

        threading.Thread(target=_load, daemon=True).start()

    # ------------------------------------------------------------- actions

    def new_hand(self):
        if self.hand is not None and not self.hand.terminal:
            raise ApiError("Une main est déjà en cours.")
        self._maybe_reload_model()
        self.button = 1 - self.button
        self.hand_id += 1
        self.hand = HeadsUpHand(self.rng, button=self.button)
        self.hand_recorded = False
        start = self._snapshot()  # blinds postées, aucune action encore jouée
        events = self._advance_ai()
        self._record_if_over()
        state = self._state(events)
        state["start"] = start
        return state

    def human_action(self, action):
        h = self.hand
        if h is None or h.terminal:
            raise ApiError("Aucune main en cours — clique sur « Nouvelle main ».")
        if h.to_act != HUMAN:
            raise ApiError("Ce n'est pas ton tour.")
        if action not in h.legal_actions():
            raise ApiError("Action illégale.")
        events = self._apply(HUMAN, action)
        events += self._advance_ai()
        self._record_if_over()
        return self._state(events)

    def current_state(self):
        return self._state([])

    # ------------------------------------------------------------- interne

    def _snapshot(self):
        """État visuel de la table à cet instant (jetons en centaines de BB via bb())."""
        h = self.hand
        return {
            "stacks": [int(s) for s in h.stacks],
            "bets": [int(b) for b in h.bets],
            "pot": int(h.pot),
            "board": [int(c) for c in h.board],
        }

    def _committed_state(self):
        """Stacks AVANT distribution du pot + pot engagé — pour figer l'affichage
        pendant le déroulé d'un tapis (sinon on divulguerait le gagnant)."""
        h = self.hand
        pot = int(h.pot)
        stacks = [int(s) for s in h.stacks]
        if h.winner == 0:
            stacks[0] -= pot
        elif h.winner == 1:
            stacks[1] -= pot
        else:  # partage
            stacks[0] -= pot // 2
            stacks[1] -= pot - pot // 2
        return stacks, pot

    def _apply(self, seat, action):
        h = self.hand
        street_before = h.street
        h.step(action)

        # Au showdown, on gèle l'état « engagé » (pré-gains) pour ne pas révéler
        # le gagnant avant la bannière de résultat.
        if h.terminal and h.showdown:
            stacks, pot = self._committed_state()
            snap = {"stacks": stacks, "bets": [int(b) for b in h.bets],
                    "pot": pot, "board": [int(c) for c in h.board]}
        else:
            snap = self._snapshot()

        events = [{
            "who": "toi" if seat == HUMAN else "ia",
            "action": int(action),
            "name": ACTION_NAMES[action],
            "street": int(street_before),
            **snap,
        }]

        # Révèle les streets nouvellement ouvertes : avance normale (une seule)
        # OU déroulé d'un tapis payé (flop + turn + river d'un coup).
        for st in range(street_before + 1, h.street + 1):
            n = [0, 3, 4, 5][st]
            board = [int(c) for c in h.full_board[:n]]
            if h.terminal:  # tapis : pot plein, stacks engagés
                stacks, pot = self._committed_state()
            else:           # entre deux tours d'enchères : mises remises à zéro
                stacks, pot = [int(s) for s in h.stacks], int(h.pot)
            events.append({"who": "table", "street": int(st), "board": board,
                           "stacks": stacks, "bets": [0, 0], "pot": pot})
        return events

    def _advance_ai(self):
        events = []
        while not self.hand.terminal and self.hand.to_act == AI:
            events += self._apply(AI, self.ai.act(self.hand, AI))
        return events

    def _record_if_over(self):
        h = self.hand
        if h is None or not h.terminal or self.hand_recorded:
            return
        self.hand_recorded = True
        payoff_bb = h.payoffs[HUMAN] / game.BB
        self.history.append(round(payoff_bb, 2))
        if h.winner == HUMAN:
            self.biggest_pot = max(self.biggest_pot, h.pot)
        if h.showdown:
            if h.winner == HUMAN:
                self.showdowns["gagnes"] += 1
            elif h.winner == AI:
                self.showdowns["perdus"] += 1
            else:
                self.showdowns["partages"] += 1
        self._save_session()

    def _raise_preview(self):
        h = self.hand
        to_call = max(h.bets[AI] - h.bets[HUMAN], 0)
        pot_after_call = h.pot + to_call

        def bet_to(raise_by):
            raise_by = max(raise_by, h.last_raise, game.BB)
            add = min(to_call + raise_by, h.stacks[HUMAN])
            return int(h.bets[HUMAN] + add)

        return {
            "call": int(min(to_call, h.stacks[HUMAN])),
            "quarter": bet_to(pot_after_call // 4),
            "third": bet_to(pot_after_call // 3),
            "half": bet_to(pot_after_call // 2),
            "pot": bet_to(pot_after_call),
            "allin": int(h.bets[HUMAN] + h.stacks[HUMAN]),
        }

    def _result(self):
        h = self.hand
        if not h.terminal:
            return None
        result = {
            "winner": {HUMAN: "toi", AI: "ia", -1: "partage"}[h.winner],
            "showdown": bool(h.showdown),
            "payoff_bb": round(h.payoffs[HUMAN] / game.BB, 2),
            "pot": int(h.pot),
            "ai_cards": None,
            "categories": None,
            "full_board": [int(c) for c in h.full_board] if h.showdown else [int(c) for c in h.board],
        }
        if h.showdown:
            result["ai_cards"] = [int(c) for c in h.hole[AI]]
            result["categories"] = {
                "toi": CATEGORY_NAMES[hand_category(h.scores[HUMAN])],
                "ia": CATEGORY_NAMES[hand_category(h.scores[AI])],
            }
        return result

    def _state(self, events):
        h = self.hand
        your_turn = h is not None and not h.terminal and h.to_act == HUMAN
        total = round(sum(self.history), 2)
        return {
            "hand_id": self.hand_id,
            "your_cards": [int(c) for c in h.hole[HUMAN]] if h else [],
            "board": [int(c) for c in h.board] if h else [],
            "street": int(h.street) if h else 0,
            "street_name": STREET_NAMES[h.street] if h else "",
            "pot": int(h.pot) if h else 0,
            "stacks": [int(s) for s in h.stacks] if h else [game.START_STACK] * 2,
            "bets": [int(b) for b in h.bets] if h else [0, 0],
            "button": int(h.button) if h else self.button,
            "to_call": int(h.to_call(HUMAN)) if your_turn else 0,
            "your_turn": your_turn,
            "legal_actions": [int(a) for a in h.legal_actions()] if your_turn else [],
            "raise_preview": self._raise_preview() if your_turn else None,
            "terminal": h.terminal if h else True,
            "events": events,
            "result": self._result() if h else None,
            "session": {
                "hands": len(self.history),
                "total_bb": total,
                "bb100": round(total / len(self.history) * 100, 1) if self.history else 0.0,
                "history": self.history,
                "biggest_pot": int(self.biggest_pot),
                "showdowns": self.showdowns,
            },
        }

    # -------------------------------------------------------- persistance

    def _save_session(self):
        # Écriture atomique (tmp + replace) : un Ctrl+C en pleine sauvegarde ne
        # peut pas laisser un fichier tronqué.
        try:
            os.makedirs(os.path.dirname(SESSION_PATH), exist_ok=True)
            tmp = SESSION_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"history": self.history, "biggest_pot": self.biggest_pot,
                           "showdowns": self.showdowns, "hand_id": self.hand_id}, f)
            os.replace(tmp, SESSION_PATH)
        except OSError:
            pass

    def _load_session(self):
        try:
            with open(SESSION_PATH) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            history = data.get("history", [])
            if isinstance(history, list):
                self.history = [float(v) for v in history if isinstance(v, (int, float))]
            self.biggest_pot = int(data.get("biggest_pot", 0) or 0)
            sd = data.get("showdowns", {})
            if isinstance(sd, dict):
                for k in self.showdowns:
                    self.showdowns[k] = int(sd.get(k, 0) or 0)
            # Reprend la numérotation là où elle s'était arrêtée.
            self.hand_id = int(data.get("hand_id", len(self.history)) or 0)
        except (OSError, ValueError, TypeError, AttributeError):
            pass  # fichier corrompu : on repart d'une session vierge

    def reset_session(self):
        self.hand = None  # abandonne une éventuelle main en cours
        self.hand_recorded = False
        self.history = []
        self.biggest_pot = 0
        self.hand_id = 0
        self.showdowns = {"gagnes": 0, "perdus": 0, "partages": 0}
        self._save_session()
        return self.current_state()


def load_training_metrics():
    rows = []
    try:
        with open(METRICS_PATH, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rows.append({
                        "hands": int(row["mains"]),
                        "vs_regles": float(row["bb100_vs_regles"]),
                        "vs_call": float(row["bb100_vs_call"]),
                        "vs_aleatoire": float(row["bb100_vs_aleatoire"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue  # ligne partielle (fichier en cours d'écriture)
    except OSError:
        pass
    return rows


def load_progress(max_points=3000):
    """Ticker d'entraînement (progress.csv). Lignes partielles ignorées."""
    rows = []
    try:
        with open(PROGRESS_PATH, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rows.append({
                        "hands": int(row["mains"]),
                        "eps": float(row["epsilon"]),
                        "loss": None if row["perte"] in ("", "nan") else float(row["perte"]),
                        "bb100_recent": float(row["bb100_recent"]),
                        "speed": float(row["mains_par_s"]),
                        "eta_min": float(row["eta_min"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    if len(rows) > max_points:  # sous-échantillonne pour garder le tracé léger
        step = len(rows) / float(max_points)
        rows = [rows[int(i * step)] for i in range(max_points)] + [rows[-1]]
    return rows


def _load_csv(path, fields):
    """Lecture CSV tolérante : {nom: (colonne, conversion)} ; lignes partielles ignorées."""
    rows = []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rows.append({name: conv(row[col]) for name, (col, conv) in fields.items()})
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    return rows


def load_cfr_progress():
    rows = _load_csv(CFR_PROGRESS_PATH, {
        "iters": ("iterations", int),
        "infosets": ("situations", int),
        "speed": ("iters_par_s", float),
        "eta_min": ("eta_min", float),
    })
    return rows[-1] if rows else None  # seul le dernier état importe


def load_cfr_metrics():
    return _load_csv(CFR_METRICS_PATH, {
        "iters": ("iterations", int),
        "vs_regles": ("bb100_vs_regles", float),
        "vs_dqn": ("bb100_vs_dqn", float),
    })


def make_handler(session):
    class Handler(BaseHTTPRequestHandler):
        timeout = 15  # évite qu'un client lent bloque un thread indéfiniment

        def log_message(self, fmt, *args):  # journal silencieux
            pass

        def _send_json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in STATIC_FILES:
                name, ctype = STATIC_FILES[self.path]
                try:
                    with open(os.path.join(WEB_DIR, name), "rb") as f:
                        body = f.read()
                except OSError:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                with session.lock:
                    self._send_json(session.current_state())
            elif self.path == "/api/stats":
                self._send_json({"training": load_training_metrics()})
            elif self.path == "/api/progress":
                self._send_json({"progress": load_progress(),
                                 "training": load_training_metrics(),
                                 "cfr": load_cfr_progress(),
                                 "cfr_metrics": load_cfr_metrics()})
            else:
                self.send_error(404)

        def do_POST(self):
            try:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    raise ApiError("En-tête Content-Length invalide.")
                if length < 0 or length > 1_000_000:
                    raise ApiError("Corps de requête invalide.")
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw or b"{}")
                except ValueError:
                    raise ApiError("Corps JSON invalide.")
                if not isinstance(body, dict):
                    raise ApiError("Corps JSON invalide.")
                with session.lock:
                    if self.path == "/api/new-hand":
                        self._send_json(session.new_hand())
                    elif self.path == "/api/action":
                        act = body.get("action")
                        if not isinstance(act, int) or isinstance(act, bool):
                            raise ApiError("Action invalide.")
                        self._send_json(session.human_action(act))
                    elif self.path == "/api/reset-session":
                        self._send_json(session.reset_session())
                    else:
                        self.send_error(404)
            except ApiError as e:
                self._send_json({"error": str(e)}, status=400)
            except Exception as e:  # noqa: BLE001 — renvoyé au client pour debug
                self._send_json({"error": f"Erreur serveur : {e}"}, status=500)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Dashboard de la table de poker")
    parser.add_argument("--model", default=None,
                        help="cerveau de l'IA : .pkl (blueprint CFR) ou .npz (réseau DQN) ; "
                             "par défaut, le meilleur disponible dans models/")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--sims", type=int, default=400,
                        help="simulations Monte-Carlo par décision de l'IA")
    parser.add_argument("--temperature", type=float, default=0.004,
                        help="stratégie mixte : 0 = toujours l'action optimale, "
                             "0.004 = mélange les décisions serrées (~0,5 BB)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.model is None:
        # Le blueprint CFR (proche de l'équilibre de Nash) prime s'il existe.
        cfr_path = os.path.join(BASE_DIR, "models", "cfr_blueprint.pkl")
        dqn_path = os.path.join(BASE_DIR, "models", "model.npz")
        args.model = cfr_path if os.path.exists(cfr_path) else dqn_path

    if not os.path.exists(args.model):
        print(f"Modèle introuvable : {args.model}\nLance d'abord :  python train.py")
        sys.exit(1)

    session = GameSession(args.model, n_sims=args.sims, seed=args.seed,
                          temperature=args.temperature)
    brain = "blueprint CFR (équilibre de Nash)" if args.model.endswith(".pkl") else "réseau DQN"
    print(f"Cerveau chargé : {brain} — {args.model}")

    handler = make_handler(session)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)

    # Écoute aussi sur ::1 : sans ça, « localhost » tente l'IPv6 en premier sous
    # Windows et chaque requête subit ~2 s d'attente avant le repli sur IPv4.
    class _Server6(ThreadingHTTPServer):
        address_family = socket.AF_INET6

    try:
        server6 = _Server6(("::1", args.port), handler)
        threading.Thread(target=server6.serve_forever, daemon=True).start()
    except OSError:
        pass  # pas d'IPv6 sur cette machine : l'IPv4 suffit

    print(f"Table de poker prête : http://localhost:{args.port}  (Ctrl+C pour arrêter)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
