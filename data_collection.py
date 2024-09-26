import yfinance as yf
import pandas as pd

class FinancialDataDownloader:
    def __init__(self, tickers, start_date, end_date):
        """
        Initialize the downloader with asset tickers, start date, and end date.
        
        Parameters:
        tickers (dict): Dictionary with asset names as keys and tickers as values.
        start_date (str): Start date for the historical data (format: 'YYYY-MM-DD').
        end_date (str): End date for the historical data (format: 'YYYY-MM-DD').
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}

    def download_data(self):
        """
        Downloads the historical price data for all tickers provided during initialization.
        
        Returns:
        dict: A dictionary where keys are asset names and values are pandas DataFrames with historical data.
        """
        for name, ticker in self.tickers.items():
            print(f"Downloading data for {name} ({ticker})...")
            self.data[name] = yf.download(ticker, start=self.start_date, end=self.end_date)
        return self.data

    def show_data(self, asset_name, num_rows=5):
        """
        Displays the first few rows of the historical data for a specific asset.
        
        Parameters:
        asset_name (str): The name of the asset (must be a key in the tickers dictionary).
        num_rows (int): Number of rows to display. Default is 5.
        """
        if asset_name in self.data:
            print(f"Showing data for {asset_name}:")
            print(self.data[asset_name].head(num_rows))
        else:
            print(f"No data found for {asset_name}. Have you downloaded the data yet?")
    
    # You can add more functions later for preprocessing, plotting, etc.

