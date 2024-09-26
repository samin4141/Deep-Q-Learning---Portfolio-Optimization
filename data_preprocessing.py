import pandas as pd
import numpy as np

class DataPreprocessing:
    def __init__(self, data_dict):
        """
        Initialize the class with a dictionary of asset data.

        Parameters:
        data_dict (dict): Dictionary where keys are asset names and values are pandas DataFrames with historical data.
        """
        self.data_dict = data_dict

    def calculate_returns(self):
        """
        Calculate the daily returns for each asset in the data dictionary.
        The return is added as a new column called 'Return' in each asset's DataFrame.
        """
        for asset, df in self.data_dict.items():
            df['Return'] = df['Adj Close'].pct_change()  # Percentage change based on the 'Adj Close' column
            df.dropna(inplace=True)  # Remove the rows with NaN values after percentage change
            print(f"Returns calculated for {asset}")

    def calculate_volatility(self, window=21):
        """
        Calculate rolling volatility for each asset in the data dictionary.
        Volatility is calculated as the rolling standard deviation of returns over a specified window.
        
        Parameters:
        window (int): The rolling window size in days for calculating volatility. Default is 21 days.
        """
        for asset, df in self.data_dict.items():
            df['Volatility'] = df['Return'].rolling(window=window).std() * np.sqrt(window)  # Annualize the volatility
            df.dropna(inplace=True)  # Remove NaN values caused by rolling window
            print(f"Volatility calculated for {asset} with window = {window} days")

    def get_preprocessed_data(self):
        """
        Returns the preprocessed data (with returns and volatility) for each asset.
        
        Returns:
        dict: Dictionary where keys are asset names and values are pandas DataFrames with calculated returns and volatility.
        """
        return self.data_dict