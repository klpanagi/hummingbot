from decimal import Decimal

from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.smartliq import SmartLiquidity
from hummingbot.strategy.smartliq.smartliq_config_map import smartliq_config_map as c_map


def start(self):
    exchange = c_map.get("exchange").value.lower()
    market = c_map.get("market").value
    token = c_map.get("token").value.upper()
    order_amount = c_map.get("order_amount").value
    spread = c_map.get("spread").value / Decimal("100")
    order_refresh_time = c_map.get("order_refresh_time").value
    order_refresh_tolerance_pct = c_map.get("order_refresh_tolerance_pct").value / Decimal("100")
    volatility_interval = c_map.get("volatility_interval").value
    avg_volatility_period = c_map.get("avg_volatility_period").value
    volatility_to_spread_multiplier = c_map.get("volatility_to_spread_multiplier").value
    max_spread = c_map.get("max_spread").value / Decimal("100")
    max_order_age = c_map.get("max_order_age").value

    self._initialize_markets([(exchange, [market])])
    base, quote = market.split("-")
    market_info = MarketTradingPairTuple(self.markets[exchange], market, base, quote)
    self.market_trading_pair_tuples = [market_info]

    self.strategy = SmartLiquidity(
        exchange=exchange,
        market_info=market_info,
        token=token,
        order_amount=order_amount,
        spread=spread,
        order_refresh_time=order_refresh_time,
        order_refresh_tolerance_pct=order_refresh_tolerance_pct,
        volatility_interval=volatility_interval,
        avg_volatility_period=avg_volatility_period,
        volatility_to_spread_multiplier=volatility_to_spread_multiplier,
        max_spread=max_spread,
        max_order_age=max_order_age,
        hb_app_notification=True
    )
