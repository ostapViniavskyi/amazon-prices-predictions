from sklearn.base import TransformerMixin, BaseEstimator

DATE_COLUMN = 'ds' # date column name compatible with fbprophet
TARGET_COLUMN = 'y' # target column name compatible with fbprophet


class TargetLagsFeaturizer(TransformerMixin):
    """Add target lags as features"""
    def __init__(self, start_lag, end_lag):
        self.start_lag = start_lag
        self.end_lag = end_lag
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for lag in range(self.start_lag, self.end_lag + 1):
            X_copy['{}_lag_{}'.format(TARGET_COLUMN, lag)] = X_copy[TARGET_COLUMN].shift(lag)
        return X_copy.iloc[self.end_lag: , :]
    

class MeanTimeDataResampler(TransformerMixin):
    """Resample time series data by given frequency"""
    def __init__(self, freq='1W'):
        self.freq = freq
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_res = X.resample(self.freq, on=DATE_COLUMN).mean()
        return X_res.reset_index()
    

class FeaturesDropper(TransformerMixin):
    """Drop given features inside a Pipeline"""
    def __init__(self, features):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.features)