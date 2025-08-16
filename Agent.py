# Agent.py 
import numpy as np 
from collections import defaultdict

class Agent:
    """
    Base class for RL agents with epsilon-greedy policies.
    Supports linear decay over episodes or count-based decay.
    """
    def __init__(self, N0, num_actions, epsilon0=1.0, epsilon_min=0.05, total_episodes=600):
        self.N0 = N0
        self.num_actions = num_actions
        self.epsilon0 = epsilon0
        self.epsilon_min = epsilon_min
        self.total_episodes = total_episodes
        # Q-table initialized per state as zero vector
        self.Q = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32))

    def choose_action_egreedy(self, state, n_s_times, train):
        no = self.N0
        action = 0

        if train == True: 
            epsilon = float(no)/(no+n_s_times[state])
            if np.random.uniform(0, 1) < epsilon: 
                #action = self.action_space.sample() 
                action = np.random.randint(0, self.num_actions)
            else: 
                action = np.argmax(self.Q[state])
                # action = np.argmin(self.Q[state, :])
        else:
             action = np.argmax(self.Q[state]) 
        return action
    
    def choose_action_edecay(self, state, n_s_times, train=True, episode=None):
        """
        Epsilon-greedy selection.
        
        Estratégia:
        - Se 'train' for True e 'episode' fornecido → aplica decaimento exponencial:
            eps = max(epsilon_min, epsilon0 * (decay_rate ** episode))
        - Se 'episode' não for fornecido → aplica decaimento baseado em visitas (N(s))
        - Caso contrário (modo avaliação) → ação greedy

        Args:
            state (int): índice do estado discretizado
            n_s_times (dict): contador de visitas ao estado
            train (bool): indica se está em modo treinamento
            episode (int): episódio atual (para cálculo de decaimento)

        Returns:
            int: ação escolhida (índice)
        """
        if train:
            if episode is not None:
                decay_rate = 0.99  # Pode ajustar para 0.98 ou 0.995
                eps = max(self.epsilon_min, self.epsilon0 * (decay_rate ** episode))
            else:
                eps = float(self.N0) / (self.N0 + n_s_times.get(state, 0))
            if np.random.rand() < eps:
                return np.random.randint(self.num_actions)
            return int(np.argmax(self.Q[state]))
        else:
            return int(np.argmax(self.Q[state]))
