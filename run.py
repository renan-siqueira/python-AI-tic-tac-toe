import torch

from src.app.rl_env import TicTacToeEnv
from src.app.qlearning_agent import QLearningAgent
from src.app.replay_memory import ReplayMemory


print('GPU Available:', torch.cuda.is_available())

# Verificando a disponibilidade da GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TicTacToeEnv()
agent = QLearningAgent(env.action_space, device)
memory = ReplayMemory(1000)

num_episodes = 5000
BATCH_SIZE = 32
for episode in range(num_episodes):
    obs = env.reset()
    done = False

    while not done:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)

        memory.push((obs, action, reward, next_obs))

        # A cada 10 episódios, treine usando amostras da Replay Memory
        if len(memory) > BATCH_SIZE and episode % 10 == 0:
            experiences = memory.sample(BATCH_SIZE)
            for experience in experiences:
                exp_obs, exp_action, exp_reward, exp_next_obs = experience
                agent.learn(exp_obs, exp_action, exp_reward, exp_next_obs)

        if reward == -10:
            agent.learn(obs, action, -1, next_obs)
        else:
            agent.learn(obs, action, reward, next_obs)

        obs = next_obs

    if (episode + 1) % 500 == 0:
        print(f"Episode {episode + 1}/{num_episodes} - Exploration Rate: {agent.exploration_rate:.5f}")

print("Training complete!")

# Testando o agente
for episode in range(5):
    obs = env.reset()
    done = False
    env.render()

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if reward == 1:
            print("Agent X won!")
        elif reward == -1:
            print("Agent O won!")
        else:
            if done:
                print("Draw!")
