#!/usr/bin/env python

from datetime import datetime
# import pandas as pd
from fastquant import get_crypto_data


# '1w', '1d' (default), '1h', '1m'
def get_historic_data(pair, time_from, time_to, resolution="1m"):
    """Get historic data for crypto pairs.

    resolution: str ('1w', '1d' (default), '1h', '1m')

    """
    data = get_crypto_data(pair, time_from, time_to, resolution)
    return data


if __name__ == '__main__':
    pair = "BTC/USDT"
    time_from = "2021-10-20"
    time_to = datetime.today().strftime('%Y-%m-%d')
    resolution = "1h"

    rsi_period = 14
    rsi_upper = 70
    rsi_lower = 30

    data = get_historic_data(pair, time_from, time_to, resolution)
    print(f'Historic data [{time_from}, {time_to}] for <{pair}>: Len={len(data)}, Resolution={resolution}')
    print(data['close'])
    # rsi = backtest("rsi",
    #                data,
    #                rsi_period=rsi_period,
    #                rsi_upper=rsi_upper,
    #                rsi_lower=rsi_lower,
    #                plot=False)
    # print(f'RSI: {rsi}')
