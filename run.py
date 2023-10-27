import os

from app.rl_env import TicTacToeEnv
from app.agent import QLearningAgent


def train_agents(agent_X, agent_O, env, num_episodes):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False

        while not done:
            if env.current_player == 1:
                action = agent_X.act(obs)
            else:
                action = agent_O.act(obs)

            next_obs, reward, done, _ = env.step(action)

            if reward == -10:
                if env.current_player == 1:
                    agent_X.learn(obs, action, -1, next_obs)
                else:
                    agent_O.learn(obs, action, -1, next_obs)
            else:
                if env.current_player == 1:
                    agent_X.learn(obs, action, reward, next_obs)
                else:
                    agent_O.learn(obs, action, -reward, next_obs)

            obs = next_obs

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - Agent X Exploration Rate: {agent_X.exploration_rate:.5f} - Agent O Exploration Rate: {agent_O.exploration_rate:.5f}")

    print("Training complete!")


def test_agents(agent_X, agent_O, env, num_tests):
    for episode in range(num_tests):
        obs = env.reset()
        done = False
        env.render()

        while not done:
            if env.current_player == 1:
                action = agent_X.act(obs)
            else:
                action = agent_O.act(obs)

            obs, reward, done, _ = env.step(action)
            env.render()

            if done:
                if reward == 1:
                    print("Agent X won!")
                elif reward == -1:
                    print("Agent O won!")
                else:
                    print("Draw!")


def play_human_vs_agent(agent_X, env):
    print("You will play against Agent X. You are 'O' and the Agent is 'X'.")
    obs = env.reset()
    done = False
    env.render()
    
    while not done:
        if env.current_player == -1:  # Assuming human is player "O"
            action = int(input("Enter your move (0-8): "))
        else:
            action = agent_X.act(obs)
        
        obs, reward, done, _ = env.step(action)
        env.render()

        if done:
            if reward == 1:
                print("Agent X won!")
            elif reward == -1:
                print("You won!")
            else:
                print("Draw!")


def menu():
    print("Select an option:")
    print("1. Train agents from scratch")
    print("2. Continue training from saved model")
    print("3. Test using saved model")
    print("4. Play against a trained agent")
    choice = int(input("Enter your choice (1/2/3/4): "))
    return choice


def main():
    env = TicTacToeEnv()
    agent_X = QLearningAgent(env.action_space)
    agent_O = QLearningAgent(env.action_space)
    
    choice = menu()

    if choice == 1:
        num_episodes = 5000
        train_agents(agent_X, agent_O, env, num_episodes)

        # Save the trained agents
        agent_X.save('agent_X.pkl')
        agent_O.save('agent_O.pkl')
        print("Agents saved successfully!")
    
    elif choice == 2:
        if os.path.exists('agent_X.pkl') and os.path.exists('agent_O.pkl'):
            print("Loading saved agents...")
            agent_X.load('agent_X.pkl')
            agent_O.load('agent_O.pkl')
            print("Agents loaded successfully!")
            
            num_episodes = int(input("Enter the number of additional episodes to train: "))
            train_agents(agent_X, agent_O, env, num_episodes)
            
            # Save the continued training
            agent_X.save('agent_X.pkl')
            agent_O.save('agent_O.pkl')
            print("Agents updated and saved successfully!")
        else:
            print("No saved model found!")
    
    elif choice == 3:
        if os.path.exists('agent_X.pkl') and os.path.exists('agent_O.pkl'):
            print("Loading saved agents for testing...")
            agent_X.load('agent_X.pkl')
            agent_O.load('agent_O.pkl')
            print("Agents loaded successfully!")
            
            num_tests = 5
            test_agents(agent_X, agent_O, env, num_tests)
        else:
            print("No saved model found!")

    elif choice == 4:
        if os.path.exists('agent_X.pkl'):
            print("Loading Agent X to play against you...")
            agent_X.load('agent_X.pkl')
            play_human_vs_agent(agent_X, env)
        else:
            print("No trained model for Agent X found! Train the agent first.")

    else:
        print("Invalid choice!")

if __name__ == '__main__':
    main()
