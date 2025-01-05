import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.optimize import minimize

class SignalPreprocessor:
    def __init__(self, window_size: int = 10):
        """
        Preprocessor for curation signal data.
        
        Args:
            window_size: Number of historical data points to consider for trend analysis
        """
        self.window_size = window_size
        self.feature_means = None
        self.feature_stds = None
        self.optimal_coefficients = None
    
    def calculate_mean_reversion_time(self, data: pd.DataFrame, coefficients: np.ndarray) -> float:
        """
        Calculate empirical mean reversion time for a given set of coefficients.
        
        Args:
            data: DataFrame containing historical APR and signal data
            coefficients: Weights for combining features
            
        Returns:
            Mean reversion time metric
        """
        # Combine features using coefficients
        combined_signal = np.dot(data[['apr', 'signal_amount', 'weekly_queries']], coefficients)
        
        # Calculate deviations from mean
        mean_signal = np.mean(combined_signal)
        deviations = combined_signal - mean_signal
        
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.signbit(deviations)))[0]
        if len(zero_crossings) < 2:
            return float('inf')
        
        # Calculate average time between crossings
        reversion_times = np.diff(zero_crossings)
        return np.mean(reversion_times)
    
    def optimize_coefficients(self, historical_data: pd.DataFrame) -> np.ndarray:
        """
        Find optimal coefficients for feature combination using grid search.
        
        Args:
            historical_data: DataFrame containing historical APR and signal data
            
        Returns:
            Optimal coefficients for feature combination
        """
        def objective(coeffs):
            normalized_coeffs = coeffs / np.sum(np.abs(coeffs))  # Normalize coefficients
            return self.calculate_mean_reversion_time(historical_data, normalized_coeffs)
        
        # Initial guess: equal weights
        n_features = 3  # apr, signal_amount, weekly_queries
        initial_coeffs = np.ones(n_features) / n_features
        
        # Optimize with constraints
        bounds = [(0, None) for _ in range(n_features)]  # Non-negative coefficients
        constraint = {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1}  # Sum of absolute values = 1
        
        result = minimize(
            objective,
            initial_coeffs,
            bounds=bounds,
            constraints=constraint,
            method='SLSQP'
        )
        
        self.optimal_coefficients = result.x
        return self.optimal_coefficients
    
    def calculate_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend-based features from historical data.
        
        Args:
            data: DataFrame containing APR and signal data
            
        Returns:
            DataFrame with additional trend features
        """
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate rolling statistics with minimum periods
        min_periods = min(len(df), self.window_size)
        df['apr_ma'] = df['apr'].rolling(window=self.window_size, min_periods=min_periods).mean()
        df['apr_std'] = df['apr'].rolling(window=self.window_size, min_periods=min_periods).std()
        
        # Fill NaN values using forward fill then backward fill
        df['apr_ma'] = df['apr_ma'].ffill().bfill().fillna(df['apr'].mean())
        df['apr_std'] = df['apr_std'].ffill().bfill().fillna(df['apr'].std())
        
        # Calculate trend features
        df['apr_trend'] = df['apr'] - df['apr_ma']
        
        # Calculate momentum with NaN handling
        df['apr_momentum'] = df['apr'].pct_change(periods=min_periods).fillna(0)
        df['signal_momentum'] = df['signal_amount'].pct_change(periods=min_periods).fillna(0)
        
        return df
    
    def fit(self, historical_data: pd.DataFrame):
        """
        Fit preprocessor on historical data.
        
        Args:
            historical_data: DataFrame containing historical APR and signal data
        """
        # Calculate optimal coefficients
        self.optimize_coefficients(historical_data)
        
        # Add trend features first
        data_with_trends = self.calculate_trend_features(historical_data)
        
        # Calculate feature statistics for normalization
        features = ['apr', 'signal_amount', 'weekly_queries', 'apr_trend', 'apr_momentum', 'signal_momentum']
        self.feature_means = data_with_trends[features].mean()
        self.feature_stds = data_with_trends[features].std()
        
        # Replace any zero standard deviations with 1 to avoid division by zero
        self.feature_stds = self.feature_stds.replace(0, 1.0)
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted parameters.
        
        Args:
            data: DataFrame containing current APR and signal data
            
        Returns:
            Normalized and transformed feature array
        """
        if self.feature_means is None or self.feature_stds is None:
            # If not fitted, use simple standardization
            features = ['apr', 'signal_amount', 'weekly_queries']
            means = data[features].mean()
            stds = data[features].std().replace(0, 1.0)
            normalized_data = (data[features] - means) / stds
            return normalized_data.values.astype(np.float32)
        
        # Add trend features
        data_with_trends = self.calculate_trend_features(data)
        
        # Normalize features
        features = ['apr', 'signal_amount', 'weekly_queries', 'apr_trend', 'apr_momentum', 'signal_momentum']
        normalized_data = (data_with_trends[features] - self.feature_means) / self.feature_stds
        
        # Combine features using optimal coefficients if available
        if self.optimal_coefficients is not None:
            base_features = normalized_data[['apr', 'signal_amount', 'weekly_queries']].values
            combined_signal = np.dot(base_features, self.optimal_coefficients)
            
            # Concatenate all features
            final_features = np.column_stack([
                normalized_data.values,
                combined_signal.reshape(-1, 1)
            ])
        else:
            final_features = normalized_data.values
        
        return final_features.astype(np.float32)
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessor and transform data.
        
        Args:
            data: DataFrame containing APR and signal data
            
        Returns:
            Normalized and transformed feature array
        """
        self.fit(data)
        return self.transform(data)
