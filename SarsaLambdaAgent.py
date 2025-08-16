# SarsaLambdaAgent.py
import numpy as np
from collections import defaultdict
from Agent import Agent

class SarsaLambdaAgent(Agent):
    """
    SARSA(λ) tabular con trazas de elegibilidad (acumulativas).
    NO redefine selección de acción: usa los métodos de Agent
    (p.ej., choose_action_edecay(...)) para epsilon-decay.
    """

    def __init__(self, trace_decay, gamma, num_states, num_actions, N0,
                 epsilon0=1.0, epsilon_min=0.05, total_episodes=600):
        super().__init__(N0, num_actions, epsilon0, epsilon_min, total_episodes)
        self.gamma = float(gamma)
        self.trace_decay = float(trace_decay)
        # Trazas de elegibilidad
        self.E = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32))

    def reset_traces(self):
        """Útil si quieres reiniciar trazas al inicio de cada episodio."""
        self.E = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32))

    def update(self, state, next_state, reward, action, next_action, Nas):
        """
        Actualización SARSA(λ) con trazas acumulativas:
          δ = r + γ * Q[s',a'] - Q[s,a]
          Q[s,·] ← Q[s,·] + α * δ * E[s,·]
          E[s,·] ← γ * λ * E[s,·]
        con α = 1 / N(s,a) (robusto a cero).
        """
        # TD-error
        delta = reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]

        # Acumula traza del par (s,a)
        self.E[state][action] += 1.0

        # Tasa de aprendizaje robusta
        visits = 0
        try:
            visits = int(Nas.get(state, {}).get(action, 0))
        except Exception:
            try:
                visits = int(Nas[state][action])
            except Exception:
                visits = 0
        alpha = 1.0 / max(1, visits)

        # Asegura iterar sobre todas las claves presentes en Q o E
        all_states = set(list(self.Q.keys()) + list(self.E.keys()))
        for s in all_states:
            _ = self.Q[s]  # fuerza inicialización
            _ = self.E[s]
            self.Q[s] += alpha * delta * self.E[s]
            self.E[s] *= self.gamma * self.trace_decay
