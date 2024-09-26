import pandas as pd
import numpy as np
from DQN import DQNAgent
from data_collection import FinancialDataDownloader

tickers = {
    'Vanguard_Index_Fund': 'VFINX',
    'S&P_500_Index_Fund': '^GSPC',
    'Gold': 'GC=F',
    'Crude_Oil': 'CL=F',
    '10Y_Treasury_Bonds': '^TNX'
}

# Initialize the class with tickers and date range
data_downloader = FinancialDataDownloader(tickers=tickers, start_date='2018-01-01', end_date='2023-01-01')

# Download the data
downloaded_data = data_downloader.download_data()

# Show the data for Vanguard Index Fund (VFINX)
data_downloader.show_data('Vanguard_Index_Fund')

# Show the data for S&P 500 Index Fund
data_downloader.show_data('S&P_500_Index_Fund')

state_size = 10  # Example state (e.g., 5 assets + 5 weights)
action_size = 5  # Buy/sell/hold each of 5 assets
agent = DQNAgent(state_size, action_size)

# Simulated training loop (replace with real portfolio data)
for e in range(1000):  # 1000 episodes
    state = np.random.randn(state_size)  # Example state (replace with real market data)
    for time in range(500):  # Example time steps
        action = agent.act(state)
        next_state = np.random.randn(state_size)  # Example next state (replace with real data)
        reward = np.random.rand()  # Example reward (replace with portfolio returns)
        done = time == 499
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            agent.update_target_network()
            break
    if len(agent.memory) > 32:  # Example batch size
        agent.replay(32)
