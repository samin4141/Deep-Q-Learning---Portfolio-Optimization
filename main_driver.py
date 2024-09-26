import pandas as pd
import numpy as np
from DQN import DQNAgent
from data_collection import FinancialDataDownloader
from data_preprocessing import DataPreprocessing


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
preprocessing = DataPreprocessing(data_dict=downloaded_data)

# Calculate returns for all assets
preprocessing.calculate_returns()

# Calculate volatility for all assets (using a 21-day window for monthly volatility)
preprocessing.calculate_volatility(window=21)

# Get the preprocessed data
preprocessed_data = preprocessing.get_preprocessed_data()

# Example of how to access the data for a specific asset (e.g., Vanguard Index Fund)
print(preprocessed_data['Vanguard_Index_Fund'].head())

state_size = 10  # Example state (e.g., 5 assets + 5 weights)
action_size = 5  
agent = DQNAgent(state_size, action_size)


