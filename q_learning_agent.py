import numpy as np

class QLearningAgent:
    def __init__(self, state_bins=(8, 8, 8, 8), learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.state_bins = state_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Initialize Q-table with small random values
        self.q_table = np.random.uniform(low=-1, high=1, 
                                       size=state_bins + (2,))  # 2 actions
        
    def discretize_state(self, state):
        # Define bounds for each state dimension
        bounds = [
            [-4.8, 4.8],     # Cart Position
            [-5.0, 5.0],     # Cart Velocity
            [-0.418, 0.418], # Pole Angle
            [-5.0, 5.0]      # Pole Angular Velocity
        ]
        
        # Discretize each dimension
        discrete_state = []
        for i, (s, bound, bins) in enumerate(zip(state, bounds, self.state_bins)):
            scaling = (bins - 1) / (bound[1] - bound[0])
            discrete_val = int(np.clip((s - bound[0]) * scaling, 0, bins - 1))
            discrete_state.append(discrete_val)
            
        return tuple(discrete_state)
    
    def get_action(self, state, training=True):
        discrete_state = self.discretize_state(state)
        
        # Epsilon-greedy action selection
        if training and np.random.random() < self.epsilon:
            return np.random.choice([0, 1])
        
        return np.argmax(self.q_table[discrete_state])
    
    def update(self, state, action, reward, next_state, done):
        current_state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)
        
        # Q-learning update
        current_q = self.q_table[current_state + (action,)]
        next_max_q = np.max(self.q_table[next_state])
        
        # Q-learning formula
        new_q = current_q + self.learning_rate * (
            reward + (1 - done) * self.discount_factor * next_max_q - current_q
        )
        
        self.q_table[current_state + (action,)] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        np.save(filename, self.q_table)
    
    def load(self, filename):
        self.q_table = np.load(filename) 