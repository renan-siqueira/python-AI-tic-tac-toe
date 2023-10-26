from src.app.rl_env import TicTacToeEnv
from src.app.agent import QLearningAgent


env = TicTacToeEnv()
agent = QLearningAgent(env.action_space)

num_episodes = 5000
for episode in range(num_episodes):
    obs = env.reset()
    done = False

    while not done:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        
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
