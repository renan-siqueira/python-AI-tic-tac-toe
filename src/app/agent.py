import numpy as np


class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay_rate=0.995, min_exploration_rate=0.01):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.min_exploration_rate = min_exploration_rate
        self.Q = {}

    def act(self, observation):
        str_obs = str(observation)
        if str_obs not in self.Q:
            self.Q[str_obs] = [0] * self.action_space.n

        if np.random.uniform(0, 1) < self.exploration_rate:
            available_actions = [i for i, x in enumerate(observation.flatten()) if x == 0]
            return np.random.choice(available_actions)
        else:
            return np.argmax(self.Q[str_obs])

    def learn(self, observation, action, reward, next_observation):
        str_obs = str(observation)
        str_next_obs = str(next_observation)

        if str_next_obs not in self.Q:
            self.Q[str_next_obs] = [0] * self.action_space.n

        best_next_action = np.argmax(self.Q[str_next_obs])
        td_target = reward + self.discount_factor * self.Q[str_next_obs][best_next_action]
        td_error = td_target - self.Q[str_obs][action]

        self.Q[str_obs][action] += self.learning_rate * td_error

        self.exploration_rate *= self.exploration_decay_rate
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)
