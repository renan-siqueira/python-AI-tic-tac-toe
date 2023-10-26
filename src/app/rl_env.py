import numpy as np
import gym
from gym import spaces


class TicTacToeEnv(gym.Env):
    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.board = np.zeros((3, 3))
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int32)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        return self.board

    def step(self, action):
        row, col = divmod(action, 3)
        done = False
        reward = 0

        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            if self.check_win():
                reward = 1 if self.current_player == 1 else -1
                done = True
            elif not 0 in self.board:
                done = True
            else:
                self.current_player = -1 if self.current_player == 1 else 1
        else:
            reward = -10
            done = True

        return self.board, reward, done, {}

    def check_win(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return True
            if abs(sum(self.board[:, i])) == 3:
                return True
        if abs(self.board[0, 0] + self.board[1, 1] + self.board[2, 2]) == 3:
            return True
        if abs(self.board[0, 2] + self.board[1, 1] + self.board[2, 0]) == 3:
            return True
        return False

    def render(self, mode='human'):
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 1:
                    print('X', end=' ')
                elif self.board[i, j] == -1:
                    print('O', end=' ')
                else:
                    print('.', end=' ')
            print()

    def close(self):
        pass