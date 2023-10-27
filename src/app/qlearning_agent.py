import numpy as np
import torch
import torch.optim as optim
from .qnetwork import QNetwork


class QLearningAgent:
    def __init__(self, action_space, device, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, exploration_decay_rate=0.995, min_exploration_rate=0.01):
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.min_exploration_rate = min_exploration_rate

        # Definindo a rede neural QNetwork
        self.device = device
        self.model = QNetwork(input_dim=9, output_dim=self.action_space.n) # Jogo TicTacToe tem input_dim de 9
        self.model = self.model.to(device)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(self, observation):
        # Utiliza a rede neural para escolher a ação
        if np.random.uniform(0, 1) < self.exploration_rate:
            available_actions = [i for i, x in enumerate(observation.flatten()) if x == 0]
            return np.random.choice(available_actions)
        else:
            obs_tensor = torch.FloatTensor(observation).flatten().to(self.device)
            q_values = self.model(obs_tensor)
            # Zero nas ações inválidas (casas já ocupadas)
            for idx, value in enumerate(observation.flatten()):
                if value != 0:
                    q_values[idx] = -np.inf
            return torch.argmax(q_values).item()

    def learn(self, observation, action, reward, next_observation):
        # Converte observações para tensores
        obs_tensor = torch.FloatTensor(observation).flatten().to(self.device)
        next_obs_tensor = torch.FloatTensor(next_observation).flatten().to(self.device)

        # Pega os Q-values da observação atual
        q_values = self.model(obs_tensor)
        with torch.no_grad():
            next_q_values = self.model(next_obs_tensor)

        # Calcula o TD target
        best_next_action = torch.argmax(next_q_values).item()
        td_target = reward + self.discount_factor * next_q_values[best_next_action]

        # Calcula o erro
        loss = self.loss_fn(q_values[action], td_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Atualiza a taxa de exploração
        self.exploration_rate *= self.exploration_decay_rate
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)
