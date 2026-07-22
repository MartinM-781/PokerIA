"""Petit réseau de neurones (MLP 2 couches cachées) en NumPy pur.

Backpropagation manuelle + optimiseur Adam. Sert de Q-network :
entrée = features de l'état, sortie = valeur Q de chaque action.
"""
import numpy as np


class QNetwork:
    def __init__(self, in_dim, hidden=128, out_dim=5, seed=0, lr=3e-4):
        rng = np.random.default_rng(seed)

        def he(fan_in, fan_out):
            return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)

        self.params = {
            "W1": he(in_dim, hidden), "b1": np.zeros(hidden),
            "W2": he(hidden, hidden), "b2": np.zeros(hidden),
            "W3": he(hidden, out_dim), "b3": np.zeros(out_dim),
        }
        self.lr = lr
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}

    # ------------------------------------------------------------- inférence

    def forward(self, X):
        P = self.params
        h1 = np.maximum(X @ P["W1"] + P["b1"], 0.0)
        h2 = np.maximum(h1 @ P["W2"] + P["b2"], 0.0)
        q = h2 @ P["W3"] + P["b3"]
        return q, (X, h1, h2)

    def predict(self, X):
        """Valeurs Q, X de forme (F,) ou (B, F) → (B, n_actions)."""
        return self.forward(np.atleast_2d(np.asarray(X, dtype=np.float64)))[0]

    # ----------------------------------------------------------- entraînement

    def train_step(self, X, actions, targets):
        """Descente de gradient (MSE) sur les Q des actions jouées."""
        X = np.asarray(X, dtype=np.float64)
        batch = X.shape[0]
        q, (X_, h1, h2) = self.forward(X)
        idx = np.arange(batch)
        err = q[idx, actions] - targets
        loss = float(np.mean(err ** 2))

        dq = np.zeros_like(q)
        dq[idx, actions] = 2.0 * err / batch

        P = self.params
        g = {"W3": h2.T @ dq, "b3": dq.sum(axis=0)}
        dh2 = dq @ P["W3"].T
        dh2[h2 <= 0] = 0.0
        g["W2"] = h1.T @ dh2
        g["b2"] = dh2.sum(axis=0)
        dh1 = dh2 @ P["W2"].T
        dh1[h1 <= 0] = 0.0
        g["W1"] = X_.T @ dh1
        g["b1"] = dh1.sum(axis=0)

        norm = np.sqrt(sum(float((gi ** 2).sum()) for gi in g.values()))
        if norm > 10.0:  # écrêtage global du gradient
            g = {k: gi * (10.0 / norm) for k, gi in g.items()}

        self._adam(g)
        return loss

    def _adam(self, g, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for k in self.params:
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * g[k]
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * g[k] ** 2
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v[k] / (1 - beta2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    # ------------------------------------------------------------ sauvegarde

    def get_weights(self):
        return {k: v.copy() for k, v in self.params.items()}

    def set_weights(self, weights):
        for k in self.params:
            self.params[k] = weights[k].copy()

    def save(self, path):
        np.savez(path, **self.params)

    @classmethod
    def from_weights(cls, weights, lr=3e-4):
        in_dim, hidden = weights["W1"].shape
        out_dim = weights["W3"].shape[1]
        net = cls(in_dim, hidden, out_dim, lr=lr)
        net.set_weights(weights)
        return net

    @classmethod
    def load(cls, path, lr=3e-4):
        data = np.load(path)
        return cls.from_weights({k: data[k] for k in data.files}, lr=lr)
