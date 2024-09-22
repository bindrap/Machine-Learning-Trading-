import gym
from gym import spaces
import numpy as np
import pandas as pd
import pandas_ta as ta

class TradingEnvironment(gym.Env):
    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        self.data = self._add_technical_indicators(data)
        self.action_space = spaces.Discrete(2)  # Example: 0 = hold, 1 = buy
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32
        )
        self.current_step = 0

    def _add_technical_indicators(self, data):
        data['RSI'] = ta.rsi(data['Close'])
        
        # Calculating MACD
        macd = ta.macd(data['Close'])
        data['MACD'] = macd['MACD_12_26_9']
        data['MACD_signal'] = macd['MACDs_12_26_9']
        data['MACD_hist'] = macd['MACDh_12_26_9']

        # Calculating Bollinger Bands
        bbands = ta.bbands(data['Close'])
        data['Upper_Band'] = bbands['BBU_20_2.0'] if 'BBU_20_2.0' in bbands else None
        data['Middle_Band'] = bbands['BBM_20_2.0'] if 'BBM_20_2.0' in bbands else None
        data['Lower_Band'] = bbands['BBL_20_2.0'] if 'BBL_20_2.0' in bbands else None

        # Fill NaNs if any indicator calculation results in NaN values.
        data.fillna(0, inplace=True)

        return data

    def reset(self):
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = 0  # Define your reward logic here
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        observation = self.data.iloc[self.current_step].values
        return observation