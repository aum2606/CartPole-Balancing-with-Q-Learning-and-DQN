import gymnasium as gym
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent
from visualizer import CartPoleVisualizer
import argparse
import time

def play_episode(env, agent, visualizer, max_steps=1000):
    state, _ = env.reset()
    total_reward = 0
    done = False
    truncated = False
    steps = 0
    
    while not (done or truncated) and steps < max_steps:
        visualizer.render(state)
        if visualizer.check_quit():
            break
            
        action = agent.get_action(state, training=False)
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        # Add a small delay to make the visualization more visible
        time.sleep(0.02)
    
    return total_reward, steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['q_learning', 'dqn'])
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()
    
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    visualizer = CartPoleVisualizer()
    
    if args.algorithm == 'q_learning':
        agent = QLearningAgent()
        agent.load('best_q_learning.npy')
    else:
        agent = DQNAgent()
        agent.load('best_dqn.pth')
    
    for episode in range(args.episodes):
        reward, steps = play_episode(env, agent, visualizer)
        print(f"Episode {episode + 1}: Reward = {reward}, Steps = {steps}")
    
    visualizer.close()
    env.close()

if __name__ == "__main__":
    main() 