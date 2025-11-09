import pandas as pd
import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.freq = getattr(configs, "freq", "d")
        self.changepoint_prior_scale = float(getattr(configs, "prophet_changepoint_prior_scale", 0.05))
        self.seasonality_mode = getattr(configs, "prophet_seasonality_mode", "additive")
        self.growth = getattr(configs, "prophet_growth", "linear")
        self.n_changepoints = int(getattr(configs, "prophet_n_changepoints", 25))
        self.yearly_seasonality = getattr(configs, "prophet_yearly_seasonality", "auto")
        self.weekly_seasonality = getattr(configs, "prophet_weekly_seasonality", "auto")
        self.daily_seasonality = getattr(configs, "prophet_daily_seasonality", "auto")
        self.holidays = getattr(configs, "prophet_holidays", None)
        self.fallback_window = int(getattr(configs, "prophet_fallback_window", 5))
        self.register_parameter("dummy_param", nn.Parameter(torch.zeros(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forecast_batch(x)

    def _forecast_batch(self, x: torch.Tensor) -> torch.Tensor:
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError("prophet is required for the Prophet baseline, install it via `pip install prophet`." ) from exc

        freq = self._normalize_freq(self.freq)
        x_np = x.detach().cpu().numpy()
        forecast = np.zeros((x_np.shape[0], self.pred_len, x_np.shape[2]), dtype=x_np.dtype)

        for i in range(x_np.shape[0]):
            for j in range(x_np.shape[2]):
                series = x_np[i, :, j]
                try:
                    df = pd.DataFrame({
                        "ds": pd.date_range("2000-01-01", periods=self.seq_len, freq=freq),
                        "y": series,
                    })
                    model = Prophet(
                        changepoint_prior_scale=self.changepoint_prior_scale,
                        seasonality_mode=self.seasonality_mode,
                        growth=self.growth,
                        yearly_seasonality=self.yearly_seasonality,
                        weekly_seasonality=self.weekly_seasonality,
                        daily_seasonality=self.daily_seasonality,
                        n_changepoints=self.n_changepoints,
                    )
                    if self.holidays is not None:
                        model.holidays = self.holidays
                    model.fit(df)
                    future = model.make_future_dataframe(periods=self.pred_len, freq=freq, include_history=True)
                    pred_df = model.predict(future)
                    forecast[i, :, j] = pred_df["yhat"].values[-self.pred_len:]
                except Exception:
                    fallback = np.convolve(series, np.ones(self.fallback_window) / self.fallback_window, mode="same")
                    forecast[i, :, j] = fallback[-1]
        return torch.from_numpy(forecast).to(x.device).type_as(x)

    def _normalize_freq(self, freq: str) -> str:
        if not isinstance(freq, str):
            return "D"
        freq = freq.lower()
        mapping = {
            "h": "H",
            "t": "T",
            "min": "T",
            "m": "M",
            "d": "D",
            "b": "B",
            "w": "W",
            "s": "S",
        }
        if freq in mapping:
            return mapping[freq]
        return freq.upper()
