
import numpy as np
import pandas as pd

from hummingbot.strategy.__utils__.trailing_indicators.base_trailing_indicator import BaseTrailingIndicator


class IVolatilityIndicator(BaseTrailingIndicator):
    """IVolatilityIndicator.

    Instant Volatility Indicator
    """

    def __init__(self, sampling_length: int = 30, processing_length: int = 1):
        super().__init__(sampling_length, processing_length)

    def _indicator_calculation(self) -> float:
        prices = self._sampling_buffer.get_as_numpy_array()
        # σ^2 = Σ (r - R)^2 / m, Σ{i=1-m}
        # vol = np.sqrt(np.sum(np.square(np.diff(prices))) / prices.size)
        vol_2 = np.sum(np.square(np.diff(prices))) / prices.size
        # volatility == standard deviation of mid prices
        vol = np.sqrt(vol_2)
        return vol

    def _processing_calculation(self) -> float:
        processing_array = self._processing_buffer.get_as_numpy_array()
        # Mean of standard deviations
        return np.mean(np.nan_to_num(processing_array))


class HVolatilityIndicator(BaseTrailingIndicator):
    """HVolatilityIndicator.

    Historical volatility indicator.
    """

    def __init__(self, sampling_length: int = 30, processing_length: int = 15):
        super().__init__(sampling_length, processing_length)

    def _indicator_calculation(self) -> float:
        prices = self._sampling_buffer.get_as_numpy_array()
        if prices.size > 0:
            # Σ(log(r) - log(R), Σ{i=1-m}
            return np.var(np.diff(np.log(prices)))

    def _processing_calculation(self) -> float:
        processing_array = self._processing_buffer.get_as_numpy_array()
        if processing_array.size > 0:
            # Volatility == Standard deviation of mean of variance
            return np.sqrt(np.mean(np.nan_to_num(processing_array)))


class EWMVIndicator(BaseTrailingIndicator):
    """HVolatilityIndicator.

    Exponentially weighted moving variance indicator.

    Use to calculate the exponentially weighted variance of mid prices.
    """

    def __init__(self, sampling_length: int = 30, processing_length: int = 1):
        # if processing_length != 1:
        #     raise Exception("Exponential moving variance processing_length should be 1")
        super().__init__(sampling_length, processing_length)

    def _indicator_calculation(self) -> float:
        return pd.Series(self._sampling_buffer.get_as_numpy_array()).ewm(
            span=self._samples_length, adjust=True).var().iloc[-1]

    def _processing_calculation(self) -> float:
        # The EWMA formula does not assume a long-run average variance level.
        # Thus, the concept of volatility mean reversion is not captured
        # by the EWMA.
        # processing_array = self._processing_buffer.get_as_numpy_array()
        # return np.mean(np.nan_to_num(processing_array))
        return self._processing_buffer.get_last_value()


class EWMAIndicator(BaseTrailingIndicator):
    """EWMAIndicator.

    Exponentially weighted moving average indicator.

    Use to calculate the exponentially weighted variance of mid prices.
    """

    def __init__(self, sampling_length: int = 30, processing_length: int = 1):
        # if processing_length != 1:
        #     raise Exception("Exponential moving average processing_length should be 1")
        super().__init__(sampling_length, processing_length)

    def _indicator_calculation(self) -> float:
        return pd.Series(self._sampling_buffer.get_as_numpy_array()).ewm(
            span=self._samples_length, adjust=True).mean().iloc[-1]

    def _processing_calculation(self) -> float:
        # The EWMA formula does not assume a long-run average variance level.
        # Thus, the concept of volatility mean reversion is not captured
        # by the EWMA.
        # processing_array = self._processing_buffer.get_as_numpy_array()
        # return np.mean(np.nan_to_num(processing_array))
        return self._processing_buffer.get_last_value()


class EWMVolatilityIndicator(BaseTrailingIndicator):
    """HVolatilityIndicator.

    Exponentially weighted volatility indicator.

    Use to calculate the exponentially weighted volatility of mid prices.
    """

    def __init__(self, sampling_length: int = 30, processing_length: int = 1):
        # if processing_length != 1:
        #     raise Exception("Exponential moving variance processing_length should be 1")
        super().__init__(sampling_length, processing_length)

    def _indicator_calculation(self) -> float:
        return pd.Series(self._sampling_buffer.get_as_numpy_array()).ewm(
            span=self._samples_length, adjust=True).std().iloc[-1]

    def _processing_calculation(self) -> float:
        # The EWMA formula does not assume a long-run average variance level.
        # Thus, the concept of volatility mean reversion is not captured
        # by the EWMA.
        # processing_array = self._processing_buffer.get_as_numpy_array()
        # return np.mean(np.nan_to_num(processing_array))
        return self._processing_buffer.get_last_value()
