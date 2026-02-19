from typing import Dict, Optional
from dataclasses import dataclass
from typing import Dict
from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal
import enum
import numpy as np
import pandas as pd
import yfinance as yf
import pickle

# create a enum for the stock names


class StockName(enum.Enum):
    NIFTY = "nifty_data"
    SP500 = "sp500_data"
    USDINR = "usdinr_data"
    GOLD = "gold_data"
    VIX = "vix_data"
    NSE_BANK = "nse_bank_data"
    CNX_IT = "cnx_it_data"


stocks = [
    {"name": "nifty_data", "ticker": "^NSEI"},
    {"name": "sp500_data", "ticker": "^GSPC"},
    {"name": "usdinr_data", "ticker": "USDINR=X"},
    {"name": "gold_data", "ticker": "GC=F"},
    {"name": "vix_data", "ticker": "^INDIAVIX"},
    {"name": "nse_bank_data", "ticker": "^NSEBANK"},
    {"name": "cnx_it_data", "ticker": "^CNXIT"},
]


class OHLCVData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["nifty_data", "sp500_data", "usdinr_data",
                  "gold_data", "vix_data", "nse_bank_data", "cnx_it_data"]
    data: pd.DataFrame


class DataObject(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    nifty_data: OHLCVData
    sp500_data: OHLCVData
    usdinr_data: OHLCVData
    gold_data: OHLCVData
    vix_data: OHLCVData
    nse_bank_data: OHLCVData
    cnx_it_data: OHLCVData


class YahooFinanceDataLoader:
    def __init__(self, stock: Dict[str, str], use_local: bool = False):
        self.stock = stock
        self.use_local = use_local

    def load_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        return yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = df.index.tz_localize(None)
        df = df[["Close", "Open", "High", "Low", "Volume"]]
        return df

    def run(self, start: str | None, end: str | None) -> DataObject:
        data_dict = {}
        for key, ticker in self.stock.items():
            if self.use_local:
                with open(f'../../data/{key}.pkl', 'rb') as file:
                    raw_data = pickle.load(file)
            else:
                raw_data = self.load_data(ticker, start, end)
            cleaned_data = self.clean_data(raw_data)
            data_dict[key] = OHLCVData(name=key, data=cleaned_data)
        return DataObject(**data_dict)


FEATURES = [
    "rolling_sharpe_20",
    "rolling_max_dd_60",

    "realized_vol_20",
    "vol_slope_20",

    "autocorr_20",

    "rel_spx_20",
    "usdinr_vol_20",
    "rel_gold_20",
    "vix_level_20",

    "vix_slope_20",
    "trend_strength_200",

    "sector_dispersion_ratio_20",
]


@dataclass(frozen=True)
class FeatureConfig:
    sharpe_window: int = 20
    max_dd_window: int = 60
    vol_window: int = 20
    autocorr_window: int = 20
    rel_window: int = 20
    vix_window: int = 20
    trend_window: int = 200
    sector_window: int = 20
    epsilon: float = 1e-8


class FeatureEngineer:
    def __init__(self, all_data: DataObject, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.all_data = self._align_all_data(all_data)
        self.df = self.all_data.nifty_data.data.copy()
        self.div_zero_counts: Dict[str, int] = {}

    def _align_all_data(self, all_data: DataObject) -> DataObject:
        base_index = all_data.nifty_data.data.index

        def align(df: pd.DataFrame) -> pd.DataFrame:
            aligned = df.reindex(base_index).copy()
            aligned = aligned.ffill().bfill()
            return aligned

        fields = [
            "nifty_data",
            "sp500_data",
            "usdinr_data",
            "gold_data",
            "vix_data",
            "nse_bank_data",
            "cnx_it_data",
        ]
        data_dict = {}
        for field in fields:
            ohlcv = getattr(all_data, field)
            data_dict[field] = OHLCVData(
                name=ohlcv.name, data=align(ohlcv.data))
        return DataObject(**data_dict)

    def _safe_div(self, numerator: pd.Series, denominator: pd.Series, label: str) -> pd.Series:
        zero_mask = denominator == 0
        self.div_zero_counts[label] = int(zero_mask.sum())
        denom = denominator.mask(zero_mask, self.config.epsilon)
        return numerator / denom

    def get_div_zero_counts(self) -> pd.Series:
        return pd.Series(self.div_zero_counts).sort_values(ascending=False)

    def compute_sector_dispersion(self, market_vol: pd.Series) -> pd.Series:
        cfg = self.config
        nse_bank_data = self.all_data.nse_bank_data.data
        cnx_it_data = self.all_data.cnx_it_data.data

        sector_df = pd.DataFrame({
            "nse_bank": nse_bank_data["Close"],
            "cnx_it": cnx_it_data["Close"],
        })
        sector_returns = np.log(sector_df / sector_df.shift(1))
        daily_dispersion = sector_returns.std(axis=1)
        dispersion = daily_dispersion.rolling(
            cfg.sector_window, min_periods=cfg.sector_window).mean()
        # print null counts for dispersion and market_vol
        print(
            f"Null counts - dispersion: {dispersion.isnull().sum()}, market_vol: {market_vol.isnull().sum()}", sector_df.isnull().sum().sum())
        # return self._safe_div(dispersion, market_vol, "sector_dispersion_ratio_20")
        return dispersion/market_vol

    def compute_features(self) -> pd.DataFrame:
        cfg = self.config
        df = self.df
        sp500_data = self.all_data.sp500_data.data
        usdinr_data = self.all_data.usdinr_data.data
        gold_data = self.all_data.gold_data.data
        vix_data = self.all_data.vix_data.data

        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        features = pd.DataFrame(index=df.index)

        sharpe_mean = log_ret.rolling(
            cfg.sharpe_window, min_periods=cfg.sharpe_window).mean()
        sharpe_std = log_ret.rolling(
            cfg.sharpe_window, min_periods=cfg.sharpe_window).std()
        features["rolling_sharpe_20"] = self._safe_div(
            sharpe_mean, sharpe_std, "rolling_sharpe_20")

        rolling_max = df["Close"].rolling(
            cfg.max_dd_window, min_periods=cfg.max_dd_window).max()
        drawdown = df["Close"] / rolling_max - 1.0
        features["rolling_max_dd_60"] = drawdown.rolling(
            cfg.max_dd_window, min_periods=cfg.max_dd_window).min()

        features["realized_vol_20"] = np.sqrt(
            (log_ret ** 2).rolling(cfg.vol_window, min_periods=cfg.vol_window).sum())
        features["vol_slope_20"] = features["realized_vol_20"].diff()

        features["autocorr_20"] = log_ret.rolling(cfg.autocorr_window, min_periods=cfg.autocorr_window).apply(
            lambda x: x.autocorr(lag=1),
            raw=False,
        )

        features["rel_spx_20"] = (
            np.log(df["Close"] / df["Close"].shift(cfg.rel_window))
            - np.log(sp500_data["Close"] /
                     sp500_data["Close"].shift(cfg.rel_window))
        )
        features["usdinr_vol_20"] = (
            np.log(usdinr_data["Close"] / usdinr_data["Close"].shift(1))
            .rolling(cfg.vol_window, min_periods=cfg.vol_window)
            .std()
        )
        features["rel_gold_20"] = (
            np.log(df["Close"] / df["Close"].shift(cfg.rel_window))
            - np.log(gold_data["Close"] /
                     gold_data["Close"].shift(cfg.rel_window))
        )
        features["vix_level_20"] = vix_data["Close"].rolling(
            cfg.vix_window, min_periods=cfg.vix_window).mean()
        features["vix_slope_20"] = vix_data["Close"].diff().rolling(
            cfg.vix_window, min_periods=cfg.vix_window).mean()

        trend_mean = df["Close"].rolling(
            cfg.trend_window, min_periods=cfg.trend_window).mean()
        trend_std = df["Close"].rolling(
            cfg.trend_window, min_periods=cfg.trend_window).std()
        features["trend_strength_200"] = self._safe_div(
            df["Close"] - trend_mean, trend_std, "trend_strength_200")

        market_vol = log_ret.rolling(
            cfg.sector_window, min_periods=cfg.sector_window).std()
        features["sector_dispersion_ratio_20"] = self.compute_sector_dispersion(
            market_vol)

        features["Close"] = df["Close"]
        return features

    # def standardize_features(self, df: pd.DataFrame) -> pd.DataFrame:
    #     mean = df[FEATURES].mean()
    #     std = df[FEATURES].std().replace(0, np.nan)
    #     df.loc[:, FEATURES] = (df[FEATURES] - mean) / (std + 1e-8)
    #     return df

    def run(self) -> pd.DataFrame:
        features = self.compute_features()
        features = features.dropna()
        # features = self.standardize_features(features)
        self.save_output(features, '../../data/nifty_features.pkl')
        return features[FEATURES + ["Close"]]

    def save_output(self, df: pd.DataFrame, filename: str):

        with open(filename, 'wb') as file:
            pickle.dump(df, file)


class StandardScaler:
    def fit(self, df: pd.DataFrame) -> (pd.Series, pd.Series):
        mean = df.mean()
        std = df.std()
        return mean, std

    def transform(self, df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
        return (df - mean) / (std + 1e-8)

    def fit_transform(self, df: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.Series):
        mean, std = self.fit(df)
        scaled_df = self.transform(df, mean, std)
        self.save_output(scaled_df, '../../data/nifty_features_scaled.pkl')
        return scaled_df, mean, std

    def save_output(self, df: pd.DataFrame, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump({
                "mean": df.mean(),
                "std": df.std(),
                "df": df,
            }, file)


class Pipeline:
    def __init__(self, data_loader: YahooFinanceDataLoader, feature_engineer: FeatureEngineer, scaler: StandardScaler, test_mode: bool = False):
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.scaler = scaler
        self.test_mode = test_mode

    def run(self, start: str | None, end: str | None) -> pd.DataFrame:
        all_data = self.data_loader.run(start, end)
        features_df = self.feature_engineer.run()
        if self.test_mode:
            # extract the mean and std from the saved file and use it to standardize the features, instead of fitting on the current data
            with open('../../data/nifty_features_scaled.pkl', 'rb') as file:
                saved_data = pickle.load(file)
                mean = saved_data["mean"]
                std = saved_data["std"]
            scaled_features = self.scaler.transform(
                features_df[FEATURES], mean, std)
            scaled_features["Close"] = features_df["Close"]
            return scaled_features
        else:
            scaled_features, mean, std = self.scaler.fit_transform(
                features_df[FEATURES])
            scaled_features["Close"] = features_df["Close"]
            return scaled_features


SCALER_ARTIFACT_PATH = "../../data/nifty_features_scaled.pkl"
FEATURES_PATH = "../../data/nifty_features.pkl"
TRAIN_DATASET_PATH = "../../data/nifty_train_dataset.pkl"
INFER_DATASET_PATH = "../../data/nifty_infer_dataset.pkl"

# create a enum for the stock names


class StockName(enum.Enum):
    NIFTY = "nifty_data"
    SP500 = "sp500_data"
    USDINR = "usdinr_data"
    GOLD = "gold_data"
    VIX = "vix_data"
    NSE_BANK = "nse_bank_data"
    CNX_IT = "cnx_it_data"


stocks = [
    {"name": "nifty_data", "ticker": "^NSEI"},
    {"name": "sp500_data", "ticker": "^GSPC"},
    {"name": "usdinr_data", "ticker": "USDINR=X"},
    {"name": "gold_data", "ticker": "GC=F"},
    {"name": "vix_data", "ticker": "^INDIAVIX"},
    {"name": "nse_bank_data", "ticker": "^NSEBANK"},
    {"name": "cnx_it_data", "ticker": "^CNXIT"},
]


class OHLCVData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["nifty_data", "sp500_data", "usdinr_data",
                  "gold_data", "vix_data", "nse_bank_data", "cnx_it_data"]
    data: pd.DataFrame


class DataObject(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    nifty_data: OHLCVData
    sp500_data: OHLCVData
    usdinr_data: OHLCVData
    gold_data: OHLCVData
    vix_data: OHLCVData
    nse_bank_data: OHLCVData
    cnx_it_data: OHLCVData


class YahooFinanceDataLoader:
    def __init__(self, stock: Dict[str, str], use_local: bool = False):
        self.stock = stock
        self.use_local = use_local

    def load_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        return yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = df.index.tz_localize(None)
        df = df[["Close", "Open", "High", "Low", "Volume"]]
        return df

    def run(self, start: str | None, end: str | None) -> DataObject:
        data_dict = {}
        for key, ticker in self.stock.items():
            if self.use_local:
                with open(f'../../data/{key}.pkl', 'rb') as file:
                    raw_data = pickle.load(file)
            else:
                raw_data = self.load_data(ticker, start, end)
            cleaned_data = self.clean_data(raw_data)
            data_dict[key] = OHLCVData(name=key, data=cleaned_data)
        return DataObject(**data_dict)


FEATURES = [
    "rolling_sharpe_20",
    "rolling_max_dd_60",

    "realized_vol_20",
    "vol_slope_20",

    "autocorr_20",

    "rel_spx_20",
    "usdinr_vol_20",
    "rel_gold_20",
    "vix_level_20",

    "vix_slope_20",
    "trend_strength_200",

    "sector_dispersion_ratio_20",
]


@dataclass(frozen=True)
class FeatureConfig:
    sharpe_window: int = 20
    max_dd_window: int = 60
    vol_window: int = 20
    autocorr_window: int = 20
    rel_window: int = 20
    vix_window: int = 20
    trend_window: int = 200
    sector_window: int = 20
    epsilon: float = 1e-8


class DataAligner:
    def align(self, all_data: DataObject) -> DataObject:
        base_index = all_data.nifty_data.data.index

        def align_df(df: pd.DataFrame) -> pd.DataFrame:
            aligned = df.reindex(base_index).copy()
            aligned = aligned.ffill()
            return aligned

        fields = [
            "nifty_data",
            "sp500_data",
            "usdinr_data",
            "gold_data",
            "vix_data",
            "nse_bank_data",
            "cnx_it_data",
        ]
        data_dict = {}
        for field in fields:
            ohlcv = getattr(all_data, field)
            data_dict[field] = OHLCVData(
                name=ohlcv.name, data=align_df(ohlcv.data))
        return DataObject(**data_dict)


class FeatureEngineer:
    def __init__(self, config: FeatureConfig | None = None, aligner: DataAligner | None = None):
        self.config = config or FeatureConfig()
        self.aligner = aligner or DataAligner()
        self.div_zero_counts: Dict[str, int] = {}

    def _safe_div(self, numerator: pd.Series, denominator: pd.Series, label: str) -> pd.Series:
        zero_mask = denominator == 0
        self.div_zero_counts[label] = int(zero_mask.sum())
        denom = denominator.mask(zero_mask, self.config.epsilon)
        return numerator / denom

    def get_div_zero_counts(self) -> pd.Series:
        return pd.Series(self.div_zero_counts).sort_values(ascending=False)

    def compute_sector_dispersion(self, all_data: DataObject, market_vol: pd.Series) -> pd.Series:
        cfg = self.config
        nse_bank_data = all_data.nse_bank_data.data
        cnx_it_data = all_data.cnx_it_data.data

        sector_df = pd.DataFrame({
            "nse_bank": nse_bank_data["Close"],
            "cnx_it": cnx_it_data["Close"],
        })
        sector_returns = np.log(sector_df / sector_df.shift(1))
        daily_dispersion = sector_returns.std(axis=1)
        dispersion = daily_dispersion.rolling(
            cfg.sector_window, min_periods=cfg.sector_window).mean()
        return self._safe_div(dispersion, market_vol, "sector_dispersion_ratio_20")

    def compute_features(self, all_data: DataObject) -> pd.DataFrame:
        cfg = self.config
        aligned = self.aligner.align(all_data)
        df = aligned.nifty_data.data.copy()
        sp500_data = aligned.sp500_data.data
        usdinr_data = aligned.usdinr_data.data
        gold_data = aligned.gold_data.data
        vix_data = aligned.vix_data.data

        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        features = pd.DataFrame(index=df.index)

        sharpe_mean = log_ret.rolling(
            cfg.sharpe_window, min_periods=cfg.sharpe_window).mean()
        sharpe_std = log_ret.rolling(
            cfg.sharpe_window, min_periods=cfg.sharpe_window).std()
        features["rolling_sharpe_20"] = self._safe_div(
            sharpe_mean, sharpe_std, "rolling_sharpe_20")

        rolling_max = df["Close"].rolling(
            cfg.max_dd_window, min_periods=cfg.max_dd_window).max()
        drawdown = df["Close"] / rolling_max - 1.0
        features["rolling_max_dd_60"] = drawdown.rolling(
            cfg.max_dd_window, min_periods=cfg.max_dd_window).min()

        features["realized_vol_20"] = np.sqrt(
            (log_ret ** 2).rolling(cfg.vol_window, min_periods=cfg.vol_window).sum())
        features["vol_slope_20"] = features["realized_vol_20"].diff()

        features["autocorr_20"] = log_ret.rolling(cfg.autocorr_window, min_periods=cfg.autocorr_window).apply(
            lambda x: x.autocorr(lag=1),
            raw=False,
        )

        features["rel_spx_20"] = (
            np.log(df["Close"] / df["Close"].shift(cfg.rel_window))
            - np.log(sp500_data["Close"] /
                     sp500_data["Close"].shift(cfg.rel_window))
        )
        features["usdinr_vol_20"] = (
            np.log(usdinr_data["Close"] / usdinr_data["Close"].shift(1))
            .rolling(cfg.vol_window, min_periods=cfg.vol_window)
            .std()
        )
        features["rel_gold_20"] = (
            np.log(df["Close"] / df["Close"].shift(cfg.rel_window))
            - np.log(gold_data["Close"] /
                     gold_data["Close"].shift(cfg.rel_window))
        )
        features["vix_level_20"] = vix_data["Close"].rolling(
            cfg.vix_window, min_periods=cfg.vix_window).mean()
        features["vix_slope_20"] = vix_data["Close"].diff().rolling(
            cfg.vix_window, min_periods=cfg.vix_window).mean()

        trend_mean = df["Close"].rolling(
            cfg.trend_window, min_periods=cfg.trend_window).mean()
        trend_std = df["Close"].rolling(
            cfg.trend_window, min_periods=cfg.trend_window).std()
        features["trend_strength_200"] = self._safe_div(
            df["Close"] - trend_mean, trend_std, "trend_strength_200")

        market_vol = log_ret.rolling(
            cfg.sector_window, min_periods=cfg.sector_window).std()
        features["sector_dispersion_ratio_20"] = self.compute_sector_dispersion(
            aligned, market_vol)

        features["Close"] = df["Close"]
        return features

    def run(self, all_data: DataObject, save: bool = True, output_path: str = FEATURES_PATH) -> pd.DataFrame:
        features = self.compute_features(all_data)
        features = features.dropna()
        if save:
            self.save_output(features, output_path)
        return features[FEATURES + ["Close"]]

    def save_output(self, df: pd.DataFrame, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump(df, file)


class StandardScaler:
    def fit(self, df: pd.DataFrame) -> (pd.Series, pd.Series):
        mean = df.mean()
        std = df.std()
        return mean, std

    def transform(self, df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
        return (df - mean) / (std + 1e-8)

    def fit_transform(self, df: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.Series):
        mean, std = self.fit(df)
        scaled_df = self.transform(df, mean, std)
        return scaled_df, mean, std

    def save_output(self, mean: pd.Series, std: pd.Series, feature_list: list[str], filename: str, meta: Optional[dict] = None):
        payload = {
            "mean": mean,
            "std": std,
            "features": feature_list,
            "meta": meta or {},
        }
        with open(filename, 'wb') as file:
            pickle.dump(payload, file)

    def load_output(self, filename: str) -> dict:
        with open(filename, 'rb') as file:
            return pickle.load(file)


class Pipeline:
    def __init__(self, data_loader: YahooFinanceDataLoader, feature_engineer: FeatureEngineer, scaler: StandardScaler):
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.scaler = scaler

    def _make_log_return_target(self, close: pd.Series, horizon: int) -> pd.Series:
        return np.log(close.shift(-horizon) / close)

    def fit(
        self,
        start: str | None,
        end: str | None,
        label_horizon: int = 1,
        label_col: str = "target_log_ret",
        save_dataset: bool = True,
        dataset_path: str = TRAIN_DATASET_PATH,
        scaler_artifact_path: str = SCALER_ARTIFACT_PATH,
        save_features: bool = True,
    ) -> pd.DataFrame:
        all_data = self.data_loader.run(start, end)
        features_df = self.feature_engineer.run(all_data, save=save_features)
        dataset = features_df[FEATURES].copy()
        dataset[label_col] = self._make_log_return_target(
            features_df["Close"], label_horizon)
        dataset = dataset.dropna()

        scaled_df, mean, std = self.scaler.fit_transform(dataset[FEATURES])
        scaled_df[label_col] = dataset[label_col]

        self.scaler.save_output(
            mean,
            std,
            FEATURES,
            scaler_artifact_path,
            meta={"label_horizon": label_horizon, "label_col": label_col},
        )
        if save_dataset:
            with open(dataset_path, "wb") as file:
                pickle.dump(scaled_df, file)
        return scaled_df

    def transform(
        self,
        start: str | None,
        end: str | None,
        save_dataset: bool = True,
        dataset_path: str = INFER_DATASET_PATH,
        scaler_artifact_path: str = SCALER_ARTIFACT_PATH,
        save_features: bool = False,
    ) -> pd.DataFrame:
        all_data = self.data_loader.run(start, end)
        features_df = self.feature_engineer.run(all_data, save=save_features)
        artifact = self.scaler.load_output(scaler_artifact_path)
        mean = artifact["mean"]
        std = artifact["std"]
        feature_list = artifact.get("features", FEATURES)
        scaled_df = self.scaler.transform(features_df[feature_list], mean, std)
        scaled_df["Close"] = features_df["Close"]
        if save_dataset:
            with open(dataset_path, "wb") as file:
                pickle.dump(scaled_df, file)
        return scaled_df
