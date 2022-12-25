from typing import (
    Dict,
    List,
    Tuple
)

from hummingbot.core.data_type.limit_order cimport LimitOrder
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.order_tracker cimport OrderTracker

NaN = float("nan")


cdef class SmartliqOrderTracker(OrderTracker):
    SHADOW_MAKER_ORDER_KEEP_ALIVE_DURATION = 60.0 * 1

    def __init__(self, trading_pair: str):
        super().__init__()
        self._trading_pair = trading_pair

    @property
    def active_limit_orders(self) -> List[LimitOrder]:
        limit_orders = []
        for market_pair, orders_map in self.tracked_limit_orders_map.items():
            for limit_order in orders_map.values():
                if self.has_in_flight_cancel(limit_order.client_order_id):
                    continue
                elif market_pair.trading_pair.upper() != self._trading_pair.upper():
                    continue
                limit_orders.append(limit_order)
        return limit_orders
