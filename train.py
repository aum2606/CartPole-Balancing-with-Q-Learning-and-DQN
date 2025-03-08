import gymnasium as gym
import numpy as np
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent
from visualizer import CartPoleVisualizer
import argparse

def train_q_learning(episodes=1000, render=False):
    env = gym.make('CartPole-v1')
    agent = QLearningAgent()
    visualizer = CartPoleVisualizer() if render else None
    
    best_reward = 0
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            if render:
                visualizer.render(state)
                if visualizer.check_quit():
                    break
                    
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save('best_q_learning.npy')
            
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
            
    if render:
        visualizer.close()
    env.close()
    return agent

def train_dqn(episodes=1000, render=False):
    env = gym.make('CartPole-v1')
    agent = DQNAgent()
    visualizer = CartPoleVisualizer() if render else None
    
    best_reward = 0
    target_update_frequency = 10
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            if render:
                visualizer.render(state)
                if visualizer.check_quit():
                    break
                    
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
            
        if episode % target_update_frequency == 0:
            agent.update_target_network()
            
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save('best_dqn.pth')
            
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
            
    if render:
        visualizer.close()
    env.close()
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['q_learning', 'dqn'])
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    
    if args.algorithm == 'q_learning':
        train_q_learning(episodes=args.episodes, render=args.render)
    else:
        train_dqn(episodes=args.episodes, render=args.render) 