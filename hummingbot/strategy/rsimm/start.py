from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.rsimm import RSIMarketMaking
from hummingbot.strategy.rsimm.rsimm_config_map import rsimm_config_map as c_map


def start(self):
    exchange = c_map.get("exchange").value.lower()
    market = c_map.get("market").value
    token = c_map.get("token").value.upper()
    order_amount = c_map.get("order_amount").value
    rsi_period = c_map.get("rsi_period").value
    rsi_interval = c_map.get("rsi_interval").value
    rsi_overbought = c_map.get("rsi_overbought").value
    rsi_oversold = c_map.get("rsi_oversold").value

    self._initialize_markets([(exchange, [market])])
    exchange = self.markets[exchange]
    base, quote = market.split("-")
    market_info = MarketTradingPairTuple(exchange, market, base, quote)
    self.market_trading_pair_tuples = [market_info]

    self.strategy = RSIMarketMaking()
    self.strategy.init_params(
        exchange=exchange,
        market_info=market_info,
        token=token,
        order_amount=order_amount,
        rsi_period=rsi_period,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        rsi_interval=rsi_interval,
        hb_app_notification=True
    )
