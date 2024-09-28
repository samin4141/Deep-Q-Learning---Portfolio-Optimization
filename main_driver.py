import pandas as pd
import numpy as np
from DQN import DQNAgent
from data_collection import FinancialDataDownloader
from data_preprocessing import DataPreprocessing
from PortfolioEnvironment import PortfolioEnvironment
from collections import deque
import random
import os


tickers = {
    'Vanguard_Index_Fund': 'VFINX',
    'S&P_500_Index_Fund': '^GSPC',
    'Gold': 'GC=F',
    'Crude_Oil': 'CL=F',
    '10Y_Treasury_Bonds': '^TNX'
}

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
EPISODES = int(os.getenv('EPISODES', 1000))
START_DATE = os.getenv('START_DATE', '2018-01-01')
END_DATE = os.getenv('END_DATE', '2023-01-01')
WINDOW = int(os.getenv('WINDOW', 21))

# Initialize the class with tickers and date range
data_downloader = FinancialDataDownloader(tickers=tickers, start_date='2018-01-01', end_date='2023-01-01')

# Download the data
downloaded_data = data_downloader.download_data()

# Show the data for Vanguard Index Fund (VFINX)
data_downloader.show_data('Vanguard_Index_Fund')

# Show the data for S&P 500 Index Fund
data_downloader.show_data('S&P_500_Index_Fund')
preprocessing = DataPreprocessing(data_dict=downloaded_data)

# Calculate returns for all assets
preprocessing.calculate_returns()

# Calculate volatility for all assets (using a 21-day window for monthly volatility)
preprocessing.calculate_volatility(window=WINDOW)

# Get the preprocessed data
preprocessed_data = preprocessing.get_preprocessed_data()

# Example of how to access the data for a specific asset (e.g., Vanguard Index Fund)
print(preprocessed_data['Vanguard_Index_Fund'].head())

state_size = 10  # Example state (e.g., 5 assets + 5 weights)
action_size = 5  
agent = DQNAgent(state_size, action_size)

# Training DQN for Portfolio Optimization
# Initialize the portfolio environment
env = PortfolioEnvironment(preprocessed_data)
state_size = len(env._get_state())  # State size (returns and volatility for each asset)
action_size = len(tickers)  # Number of assets in the portfolio

# Initialize DQN agent
agent = DQNAgent(state_size, action_size)

# Replay memory
replay_memory = deque(maxlen=2000)

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for time_step in range(len(preprocessed_data['Vanguard_Index_Fund']) - 1):
        # Choose action using epsilon-greedy policy
        action = agent.act(state)
        
        # Execute action and get the next state and reward
        next_state, reward, done = env.step(action)
        
        # Store experience in memory
        replay_memory.append((state, action, reward, next_state, done))

        # Train the DQN agent with mini-batch from replay memory
        if len(replay_memory) > BATCH_SIZE:
            minibatch = random.sample(replay_memory, BATCH_SIZE)
            agent.replay(minibatch)

        # Move to the next state
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode: {episode}/{EPISODES}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            break

    # Update target network every few episodes
    if episode % 10 == 0:
        agent.update_target_network()


