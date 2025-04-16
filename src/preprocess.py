import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

def create_simple_features(
        df: pd.DataFrame,
        price_lag: bool = False,
        weather_lag: bool = False,
        aggregate_features: bool = True
    ) -> pd.DataFrame:
    """
    Create simple features for strawberry price forecasting:
    1. Simple lags
    2. Aggregate climate features
    3. Seasonal features
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe
        
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with added features
    """
    df_copy = df.copy()
    
    if price_lag:
        df_copy['price_lag2'] = df_copy['price'].shift(2)
        df_copy['price_lag4'] = df_copy['price'].shift(4)

    if weather_lag:
        weather_vars = ['temp', 'solarradiation', 'precip', "windspeed"]
        for var in weather_vars:
            df_copy[f'{var}_lag2'] = df_copy[var].shift(2)
            df_copy[f'{var}_lag4'] = df_copy[var].shift(4)
    
    if aggregate_features:
        df_copy['temp_sum_4w'] = df_copy['temp'].rolling(window=4).sum()
        df_copy['solar_sum_4w'] = df_copy['solarradiation'].rolling(window=4).sum()
        df_copy['precip_sum_4w'] = df_copy['precip'].rolling(window=4).sum()
    
    # week to sin/cos for seasonality
    df_copy['week_sin'] = np.sin(2 * np.pi * df_copy['week']/52)
    df_copy['week_cos'] = np.cos(2 * np.pi * df_copy['week']/52)
    
    return df_copy

def baseline_model_data(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
        ) -> tuple[TimeSeries, TimeSeries, pd.Series, pd.Series]:
    """
    Since the data has big gaps in the target, we create an abstract timeindex
    so we can run a simple baseline model.
    """
    data_new_train_copy = train_df[["price", "date"]].copy().dropna()
    data_new_test_copy = test_df[["price", "date"]].copy().dropna()

    train_len = len(data_new_train_copy)
    test_len = len(data_new_test_copy)
    train_dates = data_new_train_copy['date'].reset_index(drop=True)
    test_dates = data_new_test_copy['date'].reset_index(drop=True)

    all_prices = pd.concat(
        [data_new_train_copy["price"], data_new_test_copy["price"]],
        ignore_index=True # Creates a new default RangeIndex 0, 1, ... N-1
    )

    abstract_df = pd.DataFrame({"price": all_prices})
    abstract_train_df = abstract_df.iloc[:train_len]
    abstract_test_df = abstract_df.iloc[train_len:]

    baseline_train = TimeSeries.from_dataframe(
        abstract_train_df,
        value_cols=["price"],
    )
    baseline_test = TimeSeries.from_dataframe(
        abstract_test_df,
        value_cols=["price"],
    )

    return baseline_train, baseline_test, train_dates, test_dates



class Preprocessor:
    def __init__(self, df):
        """
        Initializes the preprocessor with a dataframe.
        """
        self.df = df.copy()
    
    def _prepare_features(self) -> pd.DataFrame:
        """
        Creates simple features and returns a new dataframe.
        """
        self.df = create_simple_features(self.df)
        self.df = self._add_date_column()
        return self.df
    
    def _add_date_column(self)  -> pd.DataFrame:
        """
        Create a 'date' column based solely on year and week.
        We fix the day to Monday.
        """
        self.df['date'] = pd.to_datetime(
            self.df['year'].astype(str) + '-W' +
            self.df['week'].astype(str).str.zfill(2) + '-1',
            format='%G-W%V-%u'
        ).dt.to_period('W').dt.to_timestamp()
        return self.df

    @staticmethod
    def scale_darts_series(
        series: list[TimeSeries],
        scaler: Scaler = Scaler()
        ) -> tuple[Scaler, list[TimeSeries]]:
        """
        Scales the Darts TimeSeries using the provided scaler.
        """
        scaler = scaler
        series_scaled = scaler.fit_transform(series)
        return scaler, series_scaled

    def split_continuous_segments(self) -> pd.DataFrame:
        """
        Splits the dataframe into segments spanning 52 weeks (from the earliest date).
        Every row is assigned a segment ID (starting at 1).
        
        Returns:
        --------
        df_segmented : pandas DataFrame
            The dataframe with an added "segment" column.
        """
        self.df = self._prepare_features()
        self.df = self._add_date_column()
        df_sorted = self.df.sort_values("date").copy()
        first_date = df_sorted["date"].min()
        df_sorted["week_diff"] = ((df_sorted["date"] - first_date).dt.days) // 7
        df_sorted["segment"] = np.minimum((df_sorted["week_diff"] // 52) + 1, 10)
        df_sorted.drop("week_diff", axis=1, inplace=True)
        return df_sorted

    def train_test_split(self, df) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and test sets based on segment IDs.
        The last two segments (as defined by the `segment` column) are allocated to the test set,
        and the remaining segments form the training set.
        
        Parameters:
        -----------
        df : pandas DataFrame
            The input dataframe, assumed to have a 'segment' column.
            
        Returns:
        --------
        train_df, test_df : tuple of pandas DataFrame
        """
        segments = sorted(df['segment'].unique())
        test_segments = segments[-2:]  # Last two segments corresponding to the last two years
        train_df = df[~df['segment'].isin(test_segments)].copy()
        test_df = df[df['segment'].isin(test_segments)].copy()
        return train_df, test_df
    
    def build_training_series(
            self, 
            df: pd.DataFrame, 
            past_covariate_cols: list[str], 
            future_covariate_cols: list[str]
        ) -> tuple[list[TimeSeries], list[TimeSeries], list[TimeSeries]]:
        """
        Builds continuous training segments as Darts TimeSeries lists.
        The target is 'price' and is created by dropping rows where price is NaN.
        Past and future covariate series are constructed from the full group (without dropping rows).

        Returns:
            training_target_series, training_past_cov_series, training_future_cov_series
        """
        training_target_series = []
        training_past_cov_series = []
        training_future_cov_series = []

        # For each segment, sort by date.
        for seg, group in df.groupby("segment"):
            group_sorted = group.sort_values("date")
            # Drop NaNs only for the target (price) series.
            group_target = group_sorted.dropna(subset=["price"])
            ts_target = TimeSeries.from_dataframe(group_target, time_col="date", value_cols="price")
            ts_past = TimeSeries.from_dataframe(group_sorted, time_col="date", value_cols=past_covariate_cols)
            ts_future = TimeSeries.from_dataframe(group_sorted, time_col="date", value_cols=future_covariate_cols)
            training_target_series.append(ts_target)
            training_past_cov_series.append(ts_past)
            training_future_cov_series.append(ts_future)

        return training_target_series, training_past_cov_series, training_future_cov_series


    def build_test_series(
            self,
            df: pd.DataFrame, 
            past_covariate_cols: list[str], 
            future_covariate_cols: list[str]
        ) -> tuple[list[TimeSeries], list[TimeSeries], list[TimeSeries]]:
        """
        Builds continuous testing segments as Darts TimeSeries lists.
        The target is 'price' and is created by dropping rows where price is NaN.
        Past and future covariate series are constructed from the full group (without dropping rows).

        Returns:
            test_target_series, test_past_cov_series, test_future_cov_series
        """
        test_target_series = []
        test_past_cov_series = []
        test_future_cov_series = []

        for seg, group in df.groupby("segment"):
            group_sorted = group.sort_values("date")
            group_target = group_sorted.dropna(subset=["price"])
            ts_target = TimeSeries.from_dataframe(group_target, time_col="date", value_cols="price")
            ts_past = TimeSeries.from_dataframe(group_sorted, time_col="date", value_cols=past_covariate_cols)
            ts_future = TimeSeries.from_dataframe(group_sorted, time_col="date", value_cols=future_covariate_cols)
            test_target_series.append(ts_target)
            test_past_cov_series.append(ts_past)
            test_future_cov_series.append(ts_future)

        return test_target_series, test_past_cov_series, test_future_cov_series
