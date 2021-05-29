import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt

from scipy.stats import norm

class VaR:
    
    def __init__(self, df):
        
        self.df = df
        self.returns = self.df.pct_change()
        self.cov_matrix = self.returns.cov()
        
    def ef_var(self, min_var_weights, max_sharpe_weights):
        
        min_var_avg_rets = self.returns.mean()
        min_var_portfolio_mean = min_var_avg_rets @ min_var_weights
        min_var_portfolio_std = np.sqrt(min_var_weights.T @ self.cov_matrix @ min_var_weights)
        min_var_std_investment = 100000 * min_var_portfolio_std
        
        print("mininum variance portfolio expected daily reutrn:", round(min_var_portfolio_mean, 3))
        print("mininum variance portfolio volatility:", round(min_var_portfolio_std, 3))

        max_sharpe_avg_rets = self.returns.mean()
        max_sharpe_portfolio_mean = max_sharpe_avg_rets @ max_sharpe_weights
        max_sharpe_portfolio_std = np.sqrt(max_sharpe_weights.T @ self.cov_matrix @ max_sharpe_weights)
        max_sharpe_std_investment = 100000 * max_sharpe_portfolio_std
        
        print("maximum_sharpe portfolio expected daily reutrn:", round(max_sharpe_portfolio_mean, 6))
        print("maximum_sharpe portfolio volatility:", round(max_sharpe_portfolio_std, 6))
        
        x = np.arange(-0.05, 0.055, 0.001)
        
        min_var_norm_dist = norm.pdf(x, min_var_portfolio_mean, min_var_portfolio_std)
        max_sharpe_norm_dist = norm.pdf(x, max_sharpe_portfolio_mean, max_sharpe_portfolio_std)
        
        plt.figure(figsize = (17,9))
        plt.plot(x, min_var_norm_dist, color='r', label = "Mininum Variance")
        plt.plot(x, max_sharpe_norm_dist, color='b', label = "Maximum Sharpe")
        plt.legend()
        plt.xlabel("Returns (%)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
        
        min_var = norm.ppf(0.05, min_var_portfolio_mean, min_var_portfolio_std)
        max_sharpe = norm.ppf(0.05, max_sharpe_portfolio_mean, max_sharpe_portfolio_std)
        
        min_var_mean_investment = 100000 * (1 + min_var_portfolio_mean)
        min_var_std_investment = 100000 * min_var_portfolio_std
        
        max_sharpe_mean_investment = 100000 * (1 + max_sharpe_portfolio_mean)
        max_sharpe_std_investment = 100000 * max_sharpe_portfolio_std
        
        min_var_cutoff = norm.ppf(0.05, min_var_mean_investment, min_var_std_investment)
        max_sharpe_cutoff = norm.ppf(0.05, max_sharpe_mean_investment, max_sharpe_std_investment)
        
        min_var_historical_var = 100000 - min_var_cutoff
        max_sharpe_historical_var = 100000 - max_sharpe_cutoff
        
        print("1 day mininum variance VaR with 95% confidence", round((100 * min_var), 3), "%")
        print("mininum variance cutoff value:", round(min_var_cutoff, 2))
        print("mininum variance historical VaR:", round(min_var_historical_var, 2))
        print("\n")
        print("1 day maximum sharpe VaR with 95% confidence", round((100 * max_sharpe), 3), "%")
        print("maximum sharpe cutoff value:", round(max_sharpe_cutoff,2))
        print("maximum sharpe historical VaR:", round(max_sharpe_historical_var, 2))
        
        min_var_array = []
        max_sharpe_array = []
        
        num_days = int(15)
        for x in range(1, num_days + 1):
            
            min_var_array.append(np.round(min_var_historical_var * np.sqrt(x),2))
            max_sharpe_array.append(np.round(max_sharpe_historical_var * np.sqrt(x),2))
        
        plt.figure(figsize = (17,9))
        plt.xlabel("Day")
        plt.ylabel("Max portfolio loss (USD)")
        plt.title("Max portfolio loss (VaR) over 15-day period")
        plt.plot(min_var_array, "r", label = "mininum variance")
        plt.plot(max_sharpe_array, "b", label = "maximum variance")    
        plt.legend()
        plt.grid(True)
        plt.show()
        
        