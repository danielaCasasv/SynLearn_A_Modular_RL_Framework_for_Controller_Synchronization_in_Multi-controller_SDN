#QLearningAgent.py
import numpy as np 
from Agent import Agent 
from collections import defaultdict

class QLearningAgent(Agent): 
    def __init__(self, gamma, num_states, num_actions, N0,
                 lr=0.1, epsilon0=1.0, epsilon_min=0.05, total_episodes=600):
        super().__init__(N0, num_actions, epsilon0, epsilon_min, total_episodes)
        self.gamma = gamma
        self.num_states = num_states
        self.lr = lr
        
    # def update(self, state, state2, reward, action, action2, Nsa): 
    #     """ 
    #     Update the action value function using the Q-Learning update. 
    #     Q(S, A) = Q(S, A) + lr(reward + (gamma * Q(S_, A_) - Q(S, A)) 
    #     Args: 
    #         prev_state: The previous state 
    #         next_state: The next state 
    #         reward: The reward for taking the respective action 
    #         prev_action: The previous action 
    #         next_action: The next action 
    #         N(s,a): number of times that a was selected for s
    #     Returns: 
    #         None 
    #     """
    #     predict = self.Q[state][action] 
    #     target = reward + self.gamma * np.max(self.Q[state2]) 
    #     self.Q[state][action] += self.lr * (target - predict)

    def update(self, state, state2, reward, action, action2, Nsa): 
        """ 
        Atualiza a Q-table com Q-Learning usando taxa de aprendizado adaptativa:
        Q(s,a) ← Q(s,a) + (1/N(s,a)) * [r + γ * max_a' Q(s',a') - Q(s,a)]

        Args: 
            state (int): estado atual (s)
            state2 (int): próximo estado (s')
            reward (float): recompensa r
            action (int): ação tomada em s (a)
            action2 (int): próxima ação (a'), não usada aqui
            Nsa (dict): número de vezes que ação a foi escolhida em s
        """
        predict = self.Q[state][action]
        target = reward + self.gamma * np.max(self.Q[state2])

        visits = Nsa[state][action]
        if visits > 0:
            lr = 1.0 / visits
        else:
            lr = self.lr  # fallback inicial

        self.Q[state][action] += lr * (target - predict)
