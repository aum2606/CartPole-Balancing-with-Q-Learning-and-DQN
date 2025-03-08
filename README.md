# CartPole Balancing with Q-Learning and DQN

This project implements CartPole balancing using both Q-Learning and Deep Q-Network (DQN) approaches with OpenAI Gymnasium and PyTorch. The simulation is visualized using Pygame.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training:
```bash
python train.py
```

3. Watch the trained agent:
```bash
python play.py
```

## Project Structure
- `train.py`: Main training script for both Q-Learning and DQN
- `dqn_agent.py`: DQN agent implementation
- `q_learning_agent.py`: Q-Learning agent implementation
- `visualizer.py`: Pygame visualization of the CartPole environment 