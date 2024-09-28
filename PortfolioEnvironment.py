import numpy as np

class PortfolioEnvironment:
    def __init__(self, data):
        self.data = data  # Preprocessed data for all assets
        self.asset_names = list(data.keys())
        self.current_step = 0
        self.done = False

    def reset(self):
        self.current_step = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        state = []
        for asset in self.asset_names:
            state.append(self.data[asset].iloc[self.current_step]['Return'])
            state.append(self.data[asset].iloc[self.current_step]['Volatility'])
        return np.array(state)

    def step(self, action):
        """
        Step function simulates the portfolio adjustment based on the action.
        - Action modifies the portfolio weights.
        - Calculates portfolio return, reward (e.g., Sharpe Ratio), and whether the episode is done.
        """
        self.current_step += 1
        if self.current_step >= len(self.data[self.asset_names[0]]):
            self.done = True
        
        # Calculate reward (e.g., portfolio return or Sharpe Ratio)
        portfolio_return = sum(
            [self.data[asset].iloc[self.current_step]['Return'] * action[i] for i, asset in enumerate(self.asset_names)]
        )
        volatility = sum(
            [self.data[asset].iloc[self.current_step]['Volatility'] * action[i] for i, asset in enumerate(self.asset_names)]
        )
        sharpe_ratio = portfolio_return / (volatility + 1e-5)  # Prevent division by zero
        reward = sharpe_ratio

        next_state = self._get_state()
        return next_state, reward, self.done