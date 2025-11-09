import numpy as np
import torch
import torch.nn as nn

from typing import Tuple


def _parse_order(configs) -> Tuple[int, int, int]:
    return (
        int(getattr(configs, "arima_order_p", 2)),
        int(getattr(configs, "arima_order_d", 0)),
        int(getattr(configs, "arima_order_q", 1)),
    )


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.order = _parse_order(configs)
        self.maxiter = int(getattr(configs, "arima_maxiter", 200))
        self.enforce_stationarity = bool(getattr(configs, "arima_enforce_stationarity", False))
        self.enforce_invertibility = bool(getattr(configs, "arima_enforce_invertibility", False))
        self.fallback_window = int(getattr(configs, "arima_fallback_window", 5))
        self.register_parameter("dummy_param", nn.Parameter(torch.zeros(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forecast = self._forecast_batch(x)
        return forecast

    def _forecast_batch(self, x: torch.Tensor) -> torch.Tensor:
        try:
            from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
            from statsmodels.tools.sm_exceptions import ConvergenceWarning
        except ImportError as exc:
            raise ImportError("statsmodels is required for the ARIMA baseline, install it via `pip install statsmodels`." ) from exc

        x_cpu = x.detach().cpu().numpy()
        forecast = np.zeros((x_cpu.shape[0], self.pred_len, x_cpu.shape[2]), dtype=x_cpu.dtype)

        import warnings

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for i in range(x_cpu.shape[0]):
            for j in range(x_cpu.shape[2]):
                series = x_cpu[i, :, j]
                try:
                    model = StatsARIMA(
                        series,
                        order=self.order,
                        enforce_stationarity=self.enforce_stationarity,
                        enforce_invertibility=self.enforce_invertibility,
                    )
                    fitted = model.fit(method_kwargs={"maxiter": self.maxiter})
                    forecast[i, :, j] = fitted.forecast(self.pred_len)
                except Exception:
                    fallback = np.convolve(series, np.ones(self.fallback_window) / self.fallback_window, mode="same")
                    forecast[i, :, j] = fallback[-1]
        forecast_tensor = torch.from_numpy(forecast).to(x.device).type_as(x)
        return forecast_tensor
