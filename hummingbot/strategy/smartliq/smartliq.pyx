# distutils: language=c++
#                    Version 2, December 2004
#
# Copyright (C) 2022 Panayiotou, Konstantinos <klpanagi@gmail.com>
# Author: Panayiotou, Konstantinos <klpanagi@gmail.com>
#
# Everyone is permitted to copy and distribute verbatim or modified
# copies of this license document, and changing it is allowed as long
# as the name is changed.
#
#            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
#   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION
#
#  0. You just DO WHAT THE FUCK YOU WANT TO.

import asyncio
import logging
from decimal import Decimal
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd

from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.client.performance import PerformanceMetrics
from hummingbot.connector.exchange_base import ExchangeBase
from hummingbot.connector.exchange_base cimport ExchangeBase
from hummingbot.connector.parrot import get_campaign_summary
from hummingbot.core.clock import Clock
from hummingbot.core.clock cimport Clock
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.rate_oracle.rate_oracle import RateOracle
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.core.utils.estimate_fee import estimate_fee
from hummingbot.core.utils import map_df_to_str
from hummingbot.logger import HummingbotLogger
from hummingbot.model.trade_fill import TradeFill
from hummingbot.strategy.__utils__.ring_buffer import RingBuffer
from hummingbot.strategy.__utils__.ring_buffer cimport RingBuffer
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.pure_market_making.inventory_skew_calculator import (
    calculate_bid_ask_ratios_from_base_asset_ratio,
)
from hummingbot.strategy.smartliq.indicators import (
    EWMVIndicator,
    EWMVolatilityIndicator,
    HVolatilityIndicator,
    IVolatilityIndicator,
)
from hummingbot.strategy.strategy_base import StrategyBase
from hummingbot.strategy.utils import order_age

from hummingbot.strategy.pure_market_making.inventory_skew_calculator cimport \
    c_calculate_bid_ask_ratios_from_base_asset_ratio
from hummingbot.strategy.pure_market_making.inventory_skew_calculator import calculate_total_order_size
from hummingbot.strategy.pure_market_making.moving_price_band import MovingPriceBand
from hummingbot.strategy.pure_market_making import InventoryCostPriceDelegate

from hummingbot.strategy.asset_price_delegate cimport AssetPriceDelegate
from hummingbot.strategy.asset_price_delegate import AssetPriceDelegate
from hummingbot.strategy.order_book_asset_price_delegate cimport OrderBookAssetPriceDelegate

from .data_types import PriceSize, Proposal
from .order_tracker import SmartliqOrderTracker


NaN = float("nan")
s_decimal_zero = Decimal(0)
s_decimal_nan = Decimal("NaN")
s_decimal_neg_one = Decimal(-1)
s_decimal_one_oh = Decimal("1.0")
s_decimal_one_hundo = Decimal("100")
smartliq_logger = None


cdef class SmartLiquidityStrategy(StrategyBase):

    @classmethod
    def logger(cls) -> HummingbotLogger:
        global smartliq_logger
        if smartliq_logger is None:
            smartliq_logger = logging.getLogger(__name__)
        return smartliq_logger

    def init_params(self,
                    market_info: MarketTradingPairTuple,
                    token: str,
                    order_amount: Decimal,
                    inventory_skew_enabled: bool = False,
                    inventory_target_base_pct: Decimal = s_decimal_zero,
                    order_refresh_time: float = 2.0,
                    order_refresh_tolerance_pct: Decimal = s_decimal_neg_one,
                    inventory_range_multiplier: Decimal = Decimal("1"),
                    volatility_price_samples: int = 2,
                    volatility_interval: double = 1.0,
                    avg_volatility_samples: int = 1,
                    volatility_to_spread_multiplier: Decimal = Decimal("10"),
                    volatility_algorithm: str = "ivol",
                    max_spread: Decimal = s_decimal_neg_one,
                    max_order_age: float = 60. * 60.,
                    status_report_interval: float = 900,
                    bid_position: int = 3,
                    ask_position: int = 3,
                    buy_volume_in_front: Decimal = Decimal("-1"),
                    sell_volume_in_front: Decimal = Decimal("-1"),
                    ignore_over_spread: Decimal = Decimal("1.0"),
                    filled_order_delay: float = 60.0,
                    should_wait_order_cancel_confirmation: bool = True,
                    bits_behind: int = 1,
                    price_type: str = "mid_price",
                    asset_price_delegate: AssetPriceDelegate = None,
                    inventory_cost_price_delegate: InventoryCostPriceDelegate = None,
                    moving_price_band: Optional[MovingPriceBand] = None,
                    hb_app_notification: bool = True):
        if moving_price_band is None:
            moving_price_band = MovingPriceBand()
        self._market_info = market_info
        self._token = token
        self._order_amount = order_amount
        self._order_refresh_time = order_refresh_time
        self._inventory_skew_enabled = inventory_skew_enabled
        self._inventory_range_multiplier = inventory_range_multiplier

        self._order_refresh_tolerance_pct = order_refresh_tolerance_pct / Decimal('100')
        self._inventory_target_base_pct = inventory_target_base_pct / Decimal('100')
        self._max_spread = max_spread / Decimal('100')
        self._ignore_over_spread = ignore_over_spread / Decimal('100')

        self._volatility_price_samples = volatility_price_samples
        self._volatility_interval = volatility_interval
        self._avg_volatility_samples = avg_volatility_samples
        self._volatility_to_spread_multiplier = volatility_to_spread_multiplier
        self._volatility_algorithm = volatility_algorithm

        self._max_order_age = max_order_age
        self._status_report_interval = status_report_interval
        self._hb_app_notification = hb_app_notification

        self._buy_position = bid_position
        self._sell_position = ask_position
        self._buy_volume_in_front = buy_volume_in_front
        self._sell_volume_in_front = sell_volume_in_front
        self._filled_order_delay = filled_order_delay
        self._should_wait_order_cancel_confirmation = should_wait_order_cancel_confirmation
        self._bits_behind = bits_behind
        self._moving_price_band = moving_price_band
        self.logger().info(self._moving_price_band)

        self._asset_price_delegate = asset_price_delegate
        self._inventory_cost_price_delegate = inventory_cost_price_delegate
        self._price_type = self.get_price_type(price_type)

        self._filled_sell_orders_count = 0
        self._all_markets_ready = False
        self._token_balances = {}
        self._last_vol_reported = 0.
        self._ev_loop = asyncio.get_event_loop()
        self._last_pnl_update = 0.
        self._last_vol = s_decimal_zero
        self._start_time = 0
        self._calculated_buy_position = 0
        self._calculated_sell_position = 0
        self._last_own_trade_price = Decimal('nan')

        self._cancel_timestamp = 0
        self._create_timestamp = 0

        self._sb_order_tracker = SmartliqOrderTracker(self.trading_pair)

        if self._volatility_algorithm == 'ivol':
            self._vol_indicator = IVolatilityIndicator(
                sampling_length=self._volatility_price_samples,
                processing_length=self._avg_volatility_samples,
            )
        elif self._volatility_algorithm == 'hvol':
            self._vol_indicator = HVolatilityIndicator(
                sampling_length=self._volatility_price_samples,
                processing_length=self._avg_volatility_samples,
            )
        elif self._volatility_algorithm == 'ewm-var':
            self._vol_indicator = EWMVIndicator(
                sampling_length=self._volatility_price_samples,
            )
        elif self._volatility_algorithm == 'ewm-vol':
            self._vol_indicator = EWMVolatilityIndicator(
                sampling_length=self._volatility_price_samples,
            )
        else:
            raise ValueError('Volatility Algorithm does not exist')

        self._hb_app = HummingbotApplication.main_application()
        self.c_add_markets([self.exchange])

    # --------------- Read-Only Properties ------------------
    @property
    def exchange(self) -> ExchangeBase:
        return self.market_info.market

    @property
    def trading_pair(self) -> str:
        return self.market_info.trading_pair

    @property
    def volatility(self) -> Decimal:
        return Decimal(self._vol_indicator.current_value)

    @property
    def mid_price(self) -> Decimal:
        return self.get_mid_price()

    @property
    def order_book(self) -> pd.DataFrame:
        return self.exchange.order_books[self.trading_pair]

    @property
    def market_info(self) -> MarketTradingPairTuple:
        return self._market_info

    @property
    def active_orders(self) -> List[LimitOrder]:
        return self._sb_order_tracker.active_limit_orders

    @property
    def active_buys(self) -> List[LimitOrder]:
        return [o for o in self.active_orders if o.is_buy]

    @property
    def active_sells(self) -> List[LimitOrder]:
        return [o for o in self.active_orders if not o.is_buy]

    @property
    def quote_token(self) -> str:
        return self.trading_pair.split("-")[1]

    @property
    def base_token(self) -> str:
        return self.trading_pair.split("-")[0]

    @property
    def moving_price_band(self) -> MovingPriceBand:
        return self._moving_price_band

    @property
    def price_type(self) -> PriceType:
        return self._price_type

    # --------------- End of Read-only Properties ----------------------

    # --------------- Dynamic Reconfigure Properties -------------------
    @property
    def order_amount(self) -> Decimal:
        return self._order_amount

    @order_amount.setter
    def order_amount(self, value: Decimal):
        self.logger().info(
            f'Updating <order_amount> parameter: '
            f'{self._order_amount} -> {value}'
        )
        self._order_amount = value

    @property
    def inventory_skew_enabled(self) -> bool:
        return self._inventory_skew_enabled

    @inventory_skew_enabled.setter
    def inventory_skew_enabled(self, value: bool):
        self.logger().info(
            f'Updating <inventory_skew_enabled> parameter: '
            f'{self._inventory_skew_enabled} -> {value}'
        )
        self._inventory_skew_enabled = value

    @property
    def inventory_target_base_pct(self) -> Decimal:
        return self._inventory_target_base_pct

    @inventory_target_base_pct.setter
    def inventory_target_base_pct(self, value: Decimal):
        self.logger().info(
            f'Updating <inventory_target_base_pct> parameter: '
            f'{self._inventory_target_base_pct * Decimal(100)} -> {value}'
        )
        self._inventory_target_base_pct = value / Decimal('100')

    @property
    def order_refresh_time(self) -> float:
        return self._order_refresh_time

    @order_refresh_time.setter
    def order_refresh_time(self, value: float):
        self.logger().info(
            f'Updating <order_refresh_time> parameter: '
            f'{self._order_refresh_time} -> {value}'
        )
        self._order_refresh_time = value

    @property
    def order_refresh_tolerance_pct(self) -> Decimal:
        return self._order_refresh_tolerance_pct

    @order_refresh_tolerance_pct.setter
    def order_refresh_tolerance_pct(self, value: Decimal):
        self.logger().info(
            f'Updating <order_refresh_tolerance_pct> parameter: '
            f'{self._order_refresh_tolerance_pct * Decimal(100)} -> {value}'
        )
        self._order_refresh_tolerance_pct = value / Decimal('100')

    @property
    def max_spread(self) -> Decimal:
        return self._max_spread

    @max_spread.setter
    def max_spread(self, value: Decimal):
        self.logger().info(
            f'Updating <max_spread> parameter: '
            f'{self._max_spread * Decimal(100)} -> {value}'
        )
        self._max_spread = value / Decimal('100')

    @property
    def max_order_age(self) -> float:
        return self._max_order_age

    @max_order_age.setter
    def max_order_age(self, value: float):
        self.logger().info(
            f'Updating <max_order_age> parameter: '
            f'{self._max_order_age} -> {value}'
        )
        self._max_order_age = value

    @property
    def volatility_price_samples(self) -> int:
        return self._volatility_price_samples

    @volatility_price_samples.setter
    def volatility_price_samples(self, value: int):
        self.logger().info(
            f'Updating <volatility_price_samples> parameter: '
            f'{self._volatility_price_samples} -> {value}'
        )
        self._volatility_price_samples = value

    @property
    def volatility_interval(self) -> double:
        return self._volatility_interval

    @volatility_interval.setter
    def volatility_interval(self, value):
        self.logger().info(
            f'Updating <volatility_interval> parameter: '
            f'{self._volatility_interval} -> {value}'
        )
        self._volatility_interval = value

    @property
    def avg_volatility_samples(self) -> int:
        return self._avg_volatility_samples

    @avg_volatility_samples.setter
    def avg_volatility_samples(self, value: int):
        self.logger().info(
            f'Updating <avg_volatility_samples> parameter: '
            f'{self._avg_volatility_samples} -> {value}'
        )
        self._avg_volatility_samples = value

    @property
    def volatility_to_spread_multiplier(self) -> Decimal:
        return self._volatility_to_spread_multiplier

    @volatility_to_spread_multiplier.setter
    def volatility_to_spread_multiplier(self, value):
        self.logger().info(
            f'Updating <volatility_to_spread_multiplier> parameter: '
            f'{self._volatility_to_spread_multiplier} -> {value}'
        )
        self._volatility_to_spread_multiplier = value

    @property
    def volatility_algorithm(self) -> str:
        return self._volatility_algorithm

    @volatility_algorithm.setter
    def volatility_algorithm(self, value: str):
        self.logger().info(
            f'Updating <volatility_algorithm> parameter: '
            f'{self._volatility_algorithm} -> {value}'
        )
        if value == 'ivol':
            self._vol_indicator = IVolatilityIndicator(
                sampling_length=self._volatility_price_samples,
                processing_length=self._avg_volatility_samples,
            )
        elif value == 'hvol':
            self._vol_indicator = HVolatilityIndicator(
                sampling_length=self._volatility_price_samples,
                processing_length=self._avg_volatility_samples,
            )
        elif value == 'ewm-var':
            self._vol_indicator = EWMVIndicator(
                sampling_length=self._volatility_price_samples,
            )
        elif value == 'ewm-vol':
            self._vol_indicator = EWMVolatilityIndicator(
                sampling_length=self._volatility_price_samples,
            )
        else:
            raise ValueError('Volatility Algorithm does not exist')
        self._volatility_algorithm = value

    @property
    def buy_position(self) -> int:
        return self._buy_position

    @buy_position.setter
    def buy_position(self, value: int):
        self.logger().info(
            f'Updating <buy_position> parameter: '
            f'{self._buy_position} -> {value}'
        )
        self._buy_position = value

    @property
    def sell_position(self) -> int:
        return self._sell_position

    @sell_position.setter
    def sell_position(self, value: int):
        self.logger().info(
            f'Updating <sell_position> parameter: '
            f'{self._sell_position} -> {value}'
        )
        self._sell_position = value

    @property
    def buy_volume_in_front(self) -> int:
        return self._buy_volume_in_front

    @buy_volume_in_front.setter
    def buy_volume_in_front(self, value: int):
        self.logger().info(
            f'Updating <buy_volume_in_front> parameter: '
            f'{self._buy_volume_in_front} -> {value}'
        )
        self._buy_volume_in_front = value

    @property
    def sell_volume_in_front(self) -> int:
        return self._sell_volume_in_front

    @sell_volume_in_front.setter
    def sell_volume_in_front(self, value: int):
        self.logger().info(
            f'Updating <sell_volume_in_front> parameter: '
            f'{self._sell_volume_in_front} -> {value}'
        )
        self._sell_volume_in_front = value

    @property
    def ignore_over_spread(self) -> Decimal:
        return self._ignore_over_spread

    @ignore_over_spread.setter
    def ignore_over_spread(self, value: Decimal):
        self.logger().info(
            f'Updating <ignore_over_spread> parameter: '
            f'{self._ignore_over_spread * Decimal(100)} -> {value}'
        )
        self._ignore_over_spread = value / Decimal('100')

    @property
    def filled_order_delay(self) -> float:
        return self._filled_order_delay

    @filled_order_delay.setter
    def filled_order_delay(self, value: float):
        self.logger().info(
            f'Updating <filled_order_delay> parameter: '
            f'{self._filled_order_delay} -> {value}'
        )
        self._filled_order_delay = value

    @property
    def bits_behind(self) -> int:
        return self._bits_behind

    @bits_behind.setter
    def bits_behind(self, value: int):
        self.logger().info(
            f'Updating <bits_behind> parameter: '
            f'{self._bits_behind} -> {value}'
        )
        self._bits_behind = value

    @property
    def moving_price_band_enabled(self) -> bool:
        return self._moving_price_band.enabled

    @moving_price_band_enabled.setter
    def moving_price_band_enabled(self, value: bool):
        self._moving_price_band.switch(value)

    @property
    def price_ceiling_pct(self) -> Decimal:
        return self._moving_price_band.price_ceiling_pct

    @price_ceiling_pct.setter
    def price_ceiling_pct(self, value: Decimal):
        self._moving_price_band.price_ceiling_pct = value
        self._moving_price_band.update(self._current_timestamp, self.get_price())

    @property
    def price_floor_pct(self) -> Decimal:
        return self._moving_price_band.price_floor_pct

    @price_floor_pct.setter
    def price_floor_pct(self, value: Decimal):
        self._moving_price_band.price_floor_pct = value
        self._moving_price_band.update(self._current_timestamp, self.get_price())

    @property
    def price_band_refresh_time(self) -> float:
        return self._moving_price_band.price_band_refresh_time

    @price_band_refresh_time.setter
    def price_band_refresh_time(self, value: Decimal):
        self._moving_price_band.price_band_refresh_time = value
        self._moving_price_band.update(self._current_timestamp, self.get_price())

    @property
    def asset_price_delegate(self) -> AssetPriceDelegate:
        return self._asset_price_delegate

    @asset_price_delegate.setter
    def asset_price_delegate(self, value):
        self._asset_price_delegate = value

    @property
    def inventory_cost_price_delegate(self) -> AssetPriceDelegate:
        return self._inventory_cost_price_delegate

    @inventory_cost_price_delegate.setter
    def inventory_cost_price_delegate(self, value):
        self._inventory_cost_price_delegate = value

    @property
    def price_type(self) -> str:
        return self._price_type

    @price_type.setter
    def price_type(self, value: str):
        self._price_type = value

    # ------------ END of Dynamic Reconfigure Parameters ---------------

    def active_orders_df(self) -> pd.DataFrame:
        """
        Return the active orders in a DataFrame.
        """
        columns = [
            "Market",
            "Side",
            "Price",
            "Spread",
            "Position",
            "Amount",
            f"Amount({self._token})",
            "Age"
        ]
        data = []
        for order in self.active_orders:
            price = self.get_price()
            spread = 0 if price == 0 else \
                abs(order.price - price) / price
            size_q = order.quantity * price
            age = order_age(order)
            # Indicates order is a paper order so 'n/a'.
            # For real orders, calculate age.
            age_txt = "n/a" if age <= 0. else \
                pd.Timestamp(age, unit='s').strftime('%H:%M:%S')
            if order.is_buy:
                position = self.order_book_index_from_price(order.price, True) + 1
            else:
                position = self.order_book_index_from_price(order.price, False) + 1
            data.append([
                order.trading_pair,
                "buy" if order.is_buy else "sell",
                float(order.price),
                f"{spread:.2%}",
                position,
                float(order.quantity),
                float(size_q),
                age_txt
            ])
        df = pd.DataFrame(data=data, columns=columns)
        df.sort_values(by=["Market", "Side"], inplace=True)
        return df

    def budget_status_df(self) -> pd.DataFrame:
        market, trading_pair, base_asset, quote_asset = self.market_info
        price = self.get_price()
        base_balance = float(market.get_balance(base_asset))
        quote_balance = float(market.get_balance(quote_asset))
        available_base_balance = float(market.get_available_balance(base_asset))
        available_quote_balance = float(market.get_available_balance(quote_asset))
        base_value = base_balance * float(price)
        total_in_quote = base_value + quote_balance
        base_ratio = base_value / total_in_quote if total_in_quote > 0 else 0
        quote_ratio = quote_balance / total_in_quote if total_in_quote > 0 else 0
        data=[
            ["", base_asset, quote_asset],
            ["Total Balance", round(base_balance, 4), round(quote_balance, 4)],
            ["Available Balance", round(available_base_balance, 4), round(available_quote_balance, 4)],
            [f"Current Value ({quote_asset})", round(base_value, 4), round(quote_balance, 4)]
        ]
        data.append(["Current %", f"{base_ratio:.1%}", f"{quote_ratio:.1%}"])
        df = pd.DataFrame(data=data)
        return df

    def market_status_data_frame(self) -> pd.DataFrame:
        markets_data = []
        markets_columns = [
            "Exchange",
            "Market",
            "Best Bid",
            "Best Ask",
            f"Ref Price ({self.price_type.name})"
        ]
        if self.price_type is PriceType.LastOwnTrade and self._last_own_trade_price.is_nan():
            markets_columns[-1] = "Ref Price (MidPrice)"
        market_books = [(self.market_info.market, self.market_info.trading_pair)]
        if type(self.asset_price_delegate) is OrderBookAssetPriceDelegate:
            market_books.append((self.asset_price_delegate.market, self.asset_price_delegate.trading_pair))
        for market, trading_pair in market_books:
            bid_price = market.get_price(trading_pair, False)
            ask_price = market.get_price(trading_pair, True)
            ref_price = float("nan")
            if market == self._market_info.market and self._inventory_cost_price_delegate is not None:
                # We're using inventory_cost, show it's price
                ref_price = self._inventory_cost_price_delegate.get_price()
                if ref_price is None:
                    ref_price = self.get_price()
            elif market == self.market_info.market and self.asset_price_delegate is None:
                ref_price = self.get_price()
            elif (
                self.asset_price_delegate is not None
                and market == self.asset_price_delegate.market
                and self.price_type is not PriceType.LastOwnTrade
            ):
                ref_price = self.asset_price_delegate.get_price_by_type(self.price_type)
            markets_data.append([
                market.display_name,
                trading_pair,
                float(bid_price),
                float(ask_price),
                float(ref_price)
            ])
        return pd.DataFrame(data=markets_data, columns=markets_columns).replace(np.nan, '', regex=True)

    async def miner_status_df(self) -> pd.DataFrame:
        """
        Return the miner status (payouts, rewards, liquidity, etc.) in a DataFrame
        """
        data = []
        g_sym = self._hb_app.client_config_map.global_token.global_token_symbol
        columns = [
            "Market",
            "Payout",
            "Reward/wk",
            "Liquidity",
            "Yield/yr",
            "Max spread"
        ]
        campaigns = await get_campaign_summary(self.exchange.display_name,
                                               [self.trading_pair])
        for market, campaign in campaigns.items():
            reward = await RateOracle.get_instance().get_value(
                amount=campaign.reward_per_wk, base_token=campaign.payout_asset
            )
            data.append([
                market,
                campaign.payout_asset,
                f"{g_sym}{reward:.0f}",
                f"{g_sym}{campaign.liquidity_usd:.0f}",
                f"{campaign.apy:.2%}",
                f"{campaign.spread_max:.2%}%"
            ])
        df = pd.DataFrame(data=data, columns=columns).replace(np.nan, '', regex=True)
        df.sort_values(by=["Market"], inplace=True)
        return df

    def assets_df(self, to_show_current_pct: bool) -> pd.DataFrame:
        market, trading_pair, base_asset, quote_asset = self.market_info
        price = self.market_info.get_mid_price()
        base_balance = float(market.get_balance(base_asset))
        quote_balance = float(market.get_balance(quote_asset))
        available_base_balance = float(market.get_available_balance(base_asset))
        available_quote_balance = float(market.get_available_balance(quote_asset))
        base_value = base_balance * float(price)
        total_in_quote = base_value + quote_balance
        base_ratio = base_value / total_in_quote if total_in_quote > 0 else 0
        quote_ratio = quote_balance / total_in_quote if total_in_quote > 0 else 0
        data=[
            ["", base_asset, quote_asset],
            ["Total Balance", round(base_balance, 4), round(quote_balance, 4)],
            ["Available Balance", round(available_base_balance, 4), round(available_quote_balance, 4)],
            [f"Current Value ({quote_asset})", round(base_value, 4), round(quote_balance, 4)]
        ]
        if to_show_current_pct:
            data.append(["Current %", f"{base_ratio:.1%}", f"{quote_ratio:.1%}"])
        df = pd.DataFrame(data=data)
        return df

    def format_status(self) -> str:
        if not self._all_markets_ready:
            return "Market connectors are not ready."
        cdef:
            list lines = []
            list warning_lines = []
        warning_lines.extend(self.network_warning([self.market_info]))

        markets_df = map_df_to_str(
            self.market_status_data_frame()
        )
        lines.extend(["", "  Markets:"] + ["    " + line for line in markets_df.to_string(index=False).split("\n")])

        assets_df = map_df_to_str(self.assets_df(not self.inventory_skew_enabled))
        # append inventory skew stats.
        if self.inventory_skew_enabled:
            inventory_skew_df = map_df_to_str(self.inventory_skew_stats_data_frame())
            assets_df = assets_df.append(inventory_skew_df)

        first_col_length = max(*assets_df[0].apply(len))
        df_lines = assets_df.to_string(index=False, header=False,
                                       formatters={0: ("{:<" + str(first_col_length) + "}").format}).split("\n")
        lines.extend(["", "  Assets:"] + ["    " + line for line in df_lines])

        # See if there're any open orders.
        if len(self.active_orders) > 0:
            df = map_df_to_str(self.active_orders_df())
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        else:
            lines.extend(["", "  No active maker orders."])

        warning_lines.extend(self.balance_warning([self._market_info]))

        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)

        return "\n".join(lines)

    def stop(self, clock: Clock):
        """stop.

        Args:
            clock (Clock): clock
        """
        pass

    cdef c_calculate_positions_from_volume_in_front(self):
        if self._sell_volume_in_front >= 0:
            for pos in range(1, len(self.order_book.snapshot[1]['amount'])):
                if sum(self.order_book.snapshot[1]['amount'][0:pos]) > self._sell_volume_in_front:
                    self._calculated_sell_position = pos + 1
                    break
            # if requested volume can't be found, go to the end of the order book
            self._calculated_sell_position = (
                self._calculated_sell_position or
                len(self.order_book.snapshot[1]['amount'])
            )
        if self._buy_volume_in_front >= 0:
            for pos in range(1, len(self.order_book.snapshot[0]['amount'])):
                if sum(self.order_book.snapshot[0]['amount'][0:pos]) > self._buy_volume_in_front:
                    self._calculated_buy_position = pos + 1
                    break
            # if requested volume can't be found, go to the end of the order book
            self._calculated_buy_position = (
                self._calculated_buy_position or
                len(self.order_book.snapshot[0]['amount'])
            )

    cdef c_update_proposal_from_volatility(self, proposal: Proposal):
        """update_proposal_from_volatility.

        Args:
            proposal (Proposal): proposal

        Returns:
            None:
        """
        cdef:
            ExchangeBase market = self._market_info.market
        ref_price = self.get_price()
        buy_spread = self.target_to_spread(proposal.buy.price)
        sell_spread = self.target_to_spread(proposal.sell.price)

        if not self.volatility.is_nan():
            # volatility applies only when it is higher than the spread setting.
            buy_spread = buy_spread + self.volatility * \
                self.volatility_to_spread_multiplier
            sell_spread = sell_spread + self.volatility * \
                self.volatility_to_spread_multiplier
        if self.max_spread > s_decimal_zero:
            buy_spread = min(buy_spread, self.max_spread)
            sell_spread = min(sell_spread, self.max_spread)
        # BUY Order ------------------------------------------------------->
        buy_price = ref_price * (Decimal("1") - buy_spread)
        buy_price = market.c_quantize_order_price(
            self.trading_pair, buy_price
        )
        # SELL Order ------------------------------------------------------>
        sell_price = ref_price * (Decimal("1") + sell_spread)
        sell_price = market.c_quantize_order_price(
            self.trading_pair, sell_price
        )
        # ------------------------------------------------------------------
        proposal.buy.price = buy_price
        proposal.sell.price = sell_price

    def cancel_active_orders(self) -> None:
        """cancel_active_orders.

        Cancel all active limit orders.

        Args:

        Returns:
            None:
        """
        for order in self.active_orders:
            self.cancel_order(
                self.market_info,
                order.client_order_id
            )

    def is_token_a_quote_token(self) -> bool:
        """is_token_a_quote_token.

        Check if self._token is a quote token
        """
        if self._token.upper() == self.quote_token.upper():
            return True
        return False

    def inventory_skew_stats_data_frame(self) -> Optional[pd.DataFrame]:
        cdef:
            ExchangeBase market = self._market_info.market

        price = self.get_price()
        base_asset_amount, quote_asset_amount = self.c_get_adjusted_available_balance(self.active_orders)
        total_order_size = calculate_total_order_size(self.order_amount, 0, 0)

        base_asset_value = base_asset_amount * price
        quote_asset_value = quote_asset_amount / price if price > s_decimal_zero else s_decimal_zero
        total_value = base_asset_amount + quote_asset_value
        total_value_in_quote = (base_asset_amount * price) + quote_asset_amount

        base_asset_ratio = (base_asset_amount / total_value
                            if total_value > s_decimal_zero
                            else s_decimal_zero)
        quote_asset_ratio = Decimal("1") - base_asset_ratio if total_value > 0 else 0
        target_base_ratio = self._inventory_target_base_pct
        inventory_range_multiplier = self._inventory_range_multiplier
        target_base_amount = (total_value * target_base_ratio
                              if price > s_decimal_zero
                              else s_decimal_zero)
        target_base_amount_in_quote = target_base_ratio * total_value_in_quote
        target_quote_amount = (1 - target_base_ratio) * total_value_in_quote

        base_asset_range = total_order_size * self._inventory_range_multiplier
        base_asset_range = min(base_asset_range, total_value * Decimal("0.5"))
        high_water_mark = target_base_amount + base_asset_range
        low_water_mark = max(target_base_amount - base_asset_range, s_decimal_zero)
        low_water_mark_ratio = (low_water_mark / total_value
                                if total_value > s_decimal_zero
                                else s_decimal_zero)
        high_water_mark_ratio = (high_water_mark / total_value
                                 if total_value > s_decimal_zero
                                 else s_decimal_zero)
        high_water_mark_ratio = min(1.0, high_water_mark_ratio)
        total_order_size_ratio = (self.order_amount * Decimal("2") / total_value
                                  if total_value > s_decimal_zero
                                  else s_decimal_zero)
        bid_ask_ratios = c_calculate_bid_ask_ratios_from_base_asset_ratio(
            float(base_asset_amount),
            float(quote_asset_amount),
            float(price),
            float(target_base_ratio),
            float(base_asset_range)
        )
        inventory_skew_df = pd.DataFrame(data=[
            [
                f"Target Value ({self.quote_token})",
                f"{target_base_amount_in_quote:.4f}",
                f"{target_quote_amount:.4f}"
            ],
            [
                "Current %",
                f"{base_asset_ratio:.1%}",
                f"{quote_asset_ratio:.1%}"
            ],
            [
                "Target %",
                f"{target_base_ratio:.1%}",
                f"{1 - target_base_ratio:.1%}"
            ],
            [
                "Inventory Range",
                f"{low_water_mark_ratio:.1%} - {high_water_mark_ratio:.1%}",
                f"{1 - high_water_mark_ratio:.1%} - {1 - low_water_mark_ratio:.1%}"
            ],
            [
                "Order Adjust %",
                f"{bid_ask_ratios.bid_ratio:.1%}",
                f"{bid_ask_ratios.ask_ratio:.1%}"
            ]
        ])
        return inventory_skew_df

    cdef tuple c_get_adjusted_available_balance(self, list orders):
        """c_get_adjusted_available_balance.

        Calculates all available balances, account for amount attributed to
        orders and reserved balance.

        Args:

        Returns:
            Tuple[Decimal, Decimal]:
        """
        cdef:
            ExchangeBase market = self._market_info.market
            object base_balance = market.c_get_available_balance(self.base_token)
            object quote_balance = market.c_get_available_balance(self.quote_token)

        for order in orders:
            if order.is_buy:
                quote_balance += order.quantity * order.price
            else:
                base_balance += order.quantity

        return base_balance, quote_balance

    cdef c_apply_inventory_skew(self, object proposal):
        cdef:
            ExchangeBase market = self.exchange
            object bid_adj_ratio
            object ask_adj_ratio
            object size

        base_balance, quote_balance = self.c_get_adjusted_available_balance(
            self.active_orders
        )

        total_order_size = calculate_total_order_size(self.order_amount, 0, 1)
        bid_ask_ratios = c_calculate_bid_ask_ratios_from_base_asset_ratio(
            float(base_balance),
            float(quote_balance),
            float(self.get_price()),
            float(self._inventory_target_base_pct),
            float(total_order_size * self._inventory_range_multiplier)
        )
        bid_adj_ratio = Decimal(bid_ask_ratios.bid_ratio)
        ask_adj_ratio = Decimal(bid_ask_ratios.ask_ratio)

        size = proposal.buy.size * bid_adj_ratio
        size = market.c_quantize_order_amount(
            self.trading_pair,
            size
        )
        proposal.buy.size = size

        size = proposal.sell.size * ask_adj_ratio
        size = market.c_quantize_order_amount(
            self.trading_pair,
            size,
            proposal.sell.price
        )
        proposal.sell.size = size

    cdef c_did_complete_buy_order(self, object order_completed_event):
        cdef:
            str order_id = order_completed_event.order_id
            limit_order_record = self._sb_order_tracker.c_get_limit_order(self.market_info, order_id)
        if limit_order_record is None:
            return

        self._filled_buy_orders_count += 1
        self._create_timestamp = self._current_timestamp + self._filled_order_delay
        self._cancel_timestamp = min(self._cancel_timestamp, self._create_timestamp)
        self._last_own_trade_price = limit_order_record.price

        msg = \
            f"({self.trading_pair}) Maker BUY order {order_id} " \
            f"({limit_order_record.quantity} {limit_order_record.base_currency} @ " \
            f"{limit_order_record.price} {limit_order_record.quote_currency}) has been completely filled."
        self.logger().info(msg)
        self.notify_hb_app_with_timestamp(msg)

    cdef c_did_complete_sell_order(self, object order_completed_event):
        cdef:
            str order_id = order_completed_event.order_id
            limit_order_record = self._sb_order_tracker.c_get_limit_order(self._market_info, order_id)
        if limit_order_record is None:
            return

        self._filled_sell_orders_count += 1
        self._create_timestamp = self._current_timestamp + self._filled_order_delay
        self._cancel_timestamp = min(self._cancel_timestamp, self._create_timestamp)
        self._last_own_trade_price = limit_order_record.price

        msg = \
            f"({self.trading_pair}) Maker SELL order {order_id} " \
            f"({limit_order_record.quantity} {limit_order_record.base_currency} @ " \
            f"{limit_order_record.price} {limit_order_record.quote_currency}) has been completely filled."
        self.logger().info(msg)
        self.notify_hb_app_with_timestamp(msg)

    def target_to_spread(self, target_price: Decimal) -> Decimal:
        """target_to_spread.

        Returns the spread of the target price, based on the current mid price
        of the market

        Args:
            target_price (Decimal): target_price

        Returns:
            Decimal:
        """
        ref_price = self.get_price()
        return abs((target_price - ref_price) / ref_price)

    def _get_order_book(self):
        return self.exchange.order_books[self.trading_pair]

    def log_positions(self, book: pd.DataFrame) -> None:
        order_idx = self.get_order_book_current_index()  # [0-.]
        formatted = self.format_order_book(book)
        orders = self.active_orders
        orders_info = []
        for order in orders:
            if order.is_buy:
                orders_info.append(
                    f'Bid (buy) order - '
                    f'price: {order.price:.6}, '
                    f'size: {order.quantity}, '
                    f'spread: {self.target_to_spread(order.price):.6%} '
                    f'position: {self.order_book_index_from_price(order.price, True) + 1}\n'
                )
            else:
                orders_info.append(
                    f'Ask (sell) order - price: {order.price:.6}, '
                    f'size: {order.quantity}, '
                    f'spread: {self.target_to_spread(order.price):.6%} '
                    f'position: {self.order_book_index_from_price(order.price, False) + 1}\n'
                )
        header = f"Positions - " \
                 f"Exchance: {self.exchange.name} | " \
                 f"Market: {self.trading_pair} | " \
                 f"Ref Price: {self.get_price():.6} | " \
                 f"Volatility: {self.volatility:.6}\n" \
                 "-" * 60 + "\n"
        text = header + '\n'.join(orders_info)
        self.logger().info(text)

    def get_order_book_current_index(self) -> Tuple[int, int]:
        book = self.exchange.order_books[self.trading_pair]
        orders = self.active_orders
        buy_index = -1
        sell_index = -1
        for order in orders:
            if order.is_buy:
                try:
                    buy_index = book.snapshot[0]['price'].loc[book.snapshot[0]['price'] ==
                                                              float(order.price)].index[0]
                except Exception as e:
                    self.logger().error(e)
                    buy_index = -1
            else:
                try:
                    sell_index = book.snapshot[1]['price'].loc[book.snapshot[1]['price'] ==
                                                               float(order.price)].index[0]
                except Exception as e:
                    self.logger().error(e)
                    sell_index = -1
        return (buy_index, sell_index)

    def order_book_price_from_index(self,
                                    index: int,
                                    is_buy: bool
                                    ) -> Decimal:
        order_book = self.exchange.order_books[self.trading_pair]
        if is_buy:
            bids = order_book.snapshot[0][['price']]
            return Decimal(bids['price'][index])
            # return Decimal(self.order_book['bid_price'][index])
        else:
            asks = order_book.snapshot[1][['price']]
            return Decimal(asks['price'][index])
            # return Decimal(self.order_book['ask_price'][index])

    def order_book_index_from_price(self, price: Decimal, is_buy: bool) -> int:
        book = self.exchange.order_books[self.trading_pair]
        if is_buy:
            try:
                index = book.snapshot[0]['price'].loc[book.snapshot[0]['price'] == float(price)].index[0]
                return index
            except Exception:
                return -1
        else:
            try:
                index = book.snapshot[1]['price'].loc[book.snapshot[1]['price'] == float(price)].index[0]
                return index
            except Exception:
                return -1

    def order_book_spread_by_index(self, index: int, is_buy: bool) -> Decimal:
        # TODO: Check that index is valid before accessing the DataFrame
        order_book = self.exchange.order_books[self.trading_pair]
        if is_buy:
            ref_price = Decimal(order_book.snapshot[0]['price'][index])
            return (ref_price - self.get_mid_price()) / self.get_mid_price()
        else:
            ref_price = Decimal(order_book.snapshot[1]['price'][index])
            return (ref_price - self.get_mid_price()) / self.get_mid_price()

    def notify_hb_app(self, msg: str):
        """notify_hb_app.

        Send a message to the hummingbot application

        Args:
            msg (str): msg
        """
        if self._hb_app_notification:
            super().notify_hb_app(msg)

    cdef object c_create_proposal_from_order_book_pos(self):
        cdef:
            ExchangeBase market = self._market_info.market
        """c_create_proposal_from_order_book_pos.

        Args:

        Returns:
            Proposal:
        """
        ob_target_buy_idx = max(
                self._buy_position,
                self._calculated_buy_position
        )  # [1,N]
        ob_target_buy_idx -= 1
        ob_target_sell_idx = max(
            self._sell_position,
            self._calculated_sell_position
        )  # [1, N]
        ob_target_sell_idx -= 1
        # Get the current indexes of buy/sell orders from order-book
        current_ob_idx = self.get_order_book_current_index()  # [0,N]

        # Initialize values for Proposal ----------------------------->
        ref_price = self.get_price()
        sell_price = buy_price = ref_price
        # -------------------------------------------------------------

        # Buy Orders ------------------------------------------------->
        if self._buy_position > 0:
            buy_price = self.order_book_price_from_index(
                ob_target_buy_idx, True
            )
        # Sell Orders ------------------------------------------------>
        if self._sell_position > 0:
            sell_price = self.order_book_price_from_index(
                ob_target_sell_idx, False
            )
        # ------------------------------------------------------------
        buy_price = market.c_quantize_order_price(
            self.trading_pair, buy_price
        )
        sell_price = market.c_quantize_order_price(
            self.trading_pair, sell_price
        )

        buy_size = market.c_quantize_order_amount(
            self.trading_pair,
            self.c_base_order_size(buy_price)
        )

        sell_size = market.c_quantize_order_amount(
            self.trading_pair,
            self.c_base_order_size(sell_price)
        )
        return Proposal(
            self.trading_pair,
            PriceSize(buy_price, buy_size),
            PriceSize(sell_price, sell_size),
        )

    cdef object c_base_order_size(self, object price):
        """c_base_order_size.

        Args:
            trading_pair (str): trading_pair
            price (Decimal): price
        """
        base, _ = self.trading_pair.split("-")
        if self._token == base:
            return self.order_amount
        if price == s_decimal_zero:
            price = self.get_price()
        return self.order_amount / price

    cdef c_apply_budget_constraint(self, object proposal):
        cdef:
            ExchangeBase market = self.exchange
            object buy_size
            object base_size
            object adjusted_amount

        base_balance, quote_balance = self.c_get_adjusted_available_balance(
            self.active_orders
        )

        buy_fee = market.c_get_fee(
            self.base_token,
            self.quote_token,
            OrderType.LIMIT,
            TradeType.BUY,
            proposal.buy.size,
            proposal.buy.price
        )
        buy_size = proposal.buy.size * proposal.buy.price * (Decimal(1) + buy_fee.percent)
        # buy_size = proposal.buy.size * proposal.buy.price

        # Adjust buy order size to use remaining balance if less than the order amount
        if quote_balance < buy_size:
            adjusted_amount = quote_balance / (proposal.buy.price * (Decimal("1") + buy_fee.percent))
            # adjusted_amount = quote_balance / proposal.buy.price
            adjusted_amount = market.c_quantize_order_amount(self.trading_pair, adjusted_amount)
            self.logger().debug(
                f'Not enough balance for BUY order '
                f'(Size: {proposal.buy.size.normalize()}, '
                f'Price: {proposal.buy.price.normalize()}), '
                f'Balance = {quote_balance} / Order size = {buy_size}, '
                f'order_amount is adjusted to {adjusted_amount}')
            proposal.buy.size = adjusted_amount
        elif quote_balance == s_decimal_zero:
            proposal.buy.size = s_decimal_zero

        sell_size = proposal.sell.size

        # Adjust sell order size to use remaining balance if less than the order amount
        if base_balance < sell_size:
            adjusted_amount = market.c_quantize_order_amount(self.trading_pair, base_balance)
            self.logger().debug(
                f'Not enough balance for SELL order '
                f'(Size: {proposal.sell.size.normalize()}, '
                f'Price: {proposal.sell.price.normalize()}), '
                f'Balance = {base_balance} / Order size = {sell_size} --> '
                f' order size is adjusted to {adjusted_amount}')
            proposal.sell.size = adjusted_amount
        elif base_balance == s_decimal_zero:
            proposal.sell.size = s_decimal_zero

    cdef c_start(self, clock: Clock, timestamp: float):
        """start.

        Args:
            clock (Clock): clock
            timestamp (float): timestamp
        """
        StrategyBase.c_start(self, clock, timestamp)
        restored_orders = self.exchange.limit_orders
        for order in restored_orders:
            self.exchange.cancel(order.trading_pair, order.client_order_id)
            self._refresh_time = self.current_timestamp + 0.1

    cdef c_execute_proposal(self, object proposal):
        cdef:
            double expiration_seconds = self._max_order_age
            bint orders_created = False
        """c_execute_proposal.

        Execute a proposal if the current timestamp is less than
        its refresh timestamp. Update the refresh timestamp.

        Args:
            proposals (List[Proposal]): proposals
        """
        maker_order_type: OrderType = self.exchange.get_maker_order_type()
        cur_orders = self.active_orders
        spread = s_decimal_zero
        if proposal.buy.size > s_decimal_zero:
            spread = self.target_to_spread(proposal.buy.price)
            self.logger().info(
                f"({proposal.market}) Creating BUY order "
                f"price: {proposal.buy}: "
                f"value: {proposal.buy.size * proposal.buy.price:.2f} "
                f"{proposal.quote()} "
                f"spread: {spread:.2%}"
            )
            self.c_buy_with_specific_market(
                self.market_info,
                proposal.buy.size,
                order_type=maker_order_type,
                price=proposal.buy.price,
                expiration_seconds=expiration_seconds
            )
            orders_created = True
        if proposal.sell.size > s_decimal_zero:
            spread = self.target_to_spread(proposal.sell.price)
            self.logger().info(
                f"({proposal.market}) Creating ASK order "
                f"price: {proposal.sell}: "
                f"value: {proposal.sell.size * proposal.sell.price:.2f} "
                f"{proposal.quote()} "
                f"spread: {spread:.2%}"
            )
            self.c_sell_with_specific_market(
                self.market_info,
                proposal.sell.size,
                order_type=maker_order_type,
                price=proposal.sell.price,
                expiration_seconds=expiration_seconds
            )
            orders_created = True
        if orders_created:
            self.set_timers()

    cdef c_update_volatility(self):
        """update_volatility

        Updates the volatility indicator with new market price data

        Args:

        Returns:
            Decimal:
        """
        if self._last_vol_reported <= self._current_timestamp - \
                self._volatility_interval:
            self._vol_indicator.add_sample(float(self.get_price()))
            if not self.volatility.is_nan() and self.volatility > 0 and \
                    float(f"{self.volatility:.6}") != float(f"{self._last_vol:.6}"):
                self.logger().info(
                    f"{self.trading_pair} "
                    f"market volatility has changed: "
                    f"{self._last_vol:.4} (V2S:"
                    f"{self._c_vol_to_spread(self._last_vol):.4}%) "
                    f"-> {self.volatility:.4} (V2S:"
                    f"{self._c_vol_to_spread(self.volatility):.4}%)"
                )
            self._last_vol_reported = self.current_timestamp
            self._last_vol = self.volatility

    cdef _c_vol_to_spread(self, volatility: Decimal):
        return volatility * self.volatility_to_spread_multiplier * Decimal(100)

    cdef object c_get_mid_price(self):
        cdef:
            AssetPriceDelegate delegate = self._asset_price_delegate
            object mid_price
        if self._asset_price_delegate is not None:
            mid_price = delegate.c_get_mid_price()
        else:
            mid_price = self.market_info.get_mid_price()
        return mid_price

    cdef c_cancel_active_orders_on_max_age_limit(self):
        """
        Cancels active non hanging orders if they are older than max age limit
        """
        cdef:
            list active_orders = self.active_orders

        if active_orders and any(order_age(o, self._current_timestamp) > \
            self._max_order_age for o in active_orders):
            self.logger().info('Cancelling orders due to max_age...')
            for order in active_orders:
                if not self.has_inflight_cancel(order.client_order_id):
                    self.c_cancel_order(
                        self.market_info,
                        order.client_order_id
                    )

    cdef bint c_is_within_tolerance(self, list current_prices, list proposal_prices):
        if len(current_prices) != len(proposal_prices):
            return False
        current_prices = sorted(current_prices)
        proposal_prices = sorted(proposal_prices)
        for current, proposal in zip(current_prices, proposal_prices):
            # if spread diff is more than the tolerance or order quantities are different, return false.
            if abs(proposal - current) / current > self._order_refresh_tolerance_pct:
                return False
        return True

    cdef c_cancel_over_tolerance_orders(self, object proposal):
        """
        Cancels active orders, checks if the order prices are within tolerance threshold
        """
        cdef:
            list active_orders = self.active_orders
            list active_buy_prices = []
            list active_sells = []
            bint to_defer_canceling = False
        if len(active_orders) == 0 or proposal is None:
            return

        if proposal is not None and \
                self._order_refresh_tolerance_pct >= 0:

            active_buy_prices = [Decimal(str(o.price)) for o in active_orders if o.is_buy]
            active_sell_prices = [Decimal(str(o.price)) for o in active_orders if not o.is_buy]
            proposal_buys = [proposal.buy.price] if proposal.buy.size > 0 else []
            proposal_sells = [proposal.sell.price] if proposal.sell.size > 0 else []

            if not self.c_is_within_tolerance(active_buy_prices, proposal_buys):
                for order in active_orders:
                    if order.is_buy:
                        self.c_cancel_order(self._market_info, order.client_order_id)

            if not self.c_is_within_tolerance(active_sell_prices, proposal_sells):
                for order in active_orders:
                    if not order.is_buy:
                        self.c_cancel_order(self._market_info, order.client_order_id)

    cdef c_ignore_orders_below_min_amount(self, object proposal):
        if proposal.buy.size * proposal.buy.price < Decimal(self._min_order_usdt):
            proposal.buy.size = s_decimal_zero
        if proposal.sell.size * proposal.sell.price < Decimal(self._min_order_usdt):
            proposal.sell.size = s_decimal_zero

    cdef bint c_to_create_orders(self, object proposal):
        cdef:
            list orders_non_cancelled = self.active_orders
            object inflight_cancels = self._sb_order_tracker.in_flight_cancels
        return (
            self._create_timestamp < self._current_timestamp
            and len(inflight_cancels) == 0
            and proposal is not None
            and len(orders_non_cancelled) == 0
        )

    cdef set_timers(self):
        cdef double next_cycle = self._current_timestamp + self._order_refresh_time
        if self._create_timestamp <= self._current_timestamp:
            self._create_timestamp = next_cycle
        if self._cancel_timestamp <= self._current_timestamp:
            self._cancel_timestamp = min(self._create_timestamp, next_cycle)

    def all_markets_ready(self):
        return all([market.ready for market in self._sb_markets])

    cdef c_tick(self, timestamp: double):
        """
        Clock tick entry point, is run every second (on normal tick setting).
        :param timestamp: current tick timestamp
        """
        StrategyBase.c_tick(self, timestamp)
        if not self._all_markets_ready:
            self._all_markets_ready = all([market.ready for market in self._sb_markets])
            if self._asset_price_delegate is not None and self._all_markets_ready:
                self._all_markets_ready = self._asset_price_delegate.ready
            if not self._all_markets_ready:
                self.logger().warning(
                    f"{self.exchange.name} is not ready. Please wait..."
                )
                return
            else:
                self.logger().info(
                    f"{self.exchange.name} is ready. Trading started."
                )
                self._start_time = self.current_timestamp

        self.c_update_volatility()
        if not self._vol_indicator.is_sampling_buffer_full:
            c_size = self._vol_indicator._sampling_buffer.get_as_numpy_array().size
            vb_size = self._volatility_price_samples
            self.logger().warning(
                'Volatility Indicator is waiting for mid-price samples'
                f' [{c_size}/{vb_size}]'
            )
            return

        proposal = None

        if self._create_timestamp <= self._current_timestamp:
            # Update the position based on the requested volume in front
            self.c_calculate_positions_from_volume_in_front()
            # Create base proposal based on order book position.
            # If the relevant parameters are set to zero, the proposal will have
            # zero amount.
            proposal = self.c_create_proposal_from_order_book_pos()
            self.c_update_proposal_from_volatility(proposal)

            if self._bits_behind > 0:
                self.c_apply_bits_behind(proposal, self._bits_behind)

            # Set ignore_over_spread parameter for this feature.
            # Ignores Orders proposals with spread > ignore_over_spread
            if self._ignore_over_spread > Decimal('0'):
                self.apply_ignore_over_spread(proposal)

            if self.moving_price_band_enabled:
                self.c_apply_moving_price_band(proposal)

            # Apply functions that modify orders size
            if self._inventory_skew_enabled:
                self.c_apply_inventory_skew(proposal)
            # Apply budget constraint, i.e. can't buy/sell more than what you have.
            self.c_apply_budget_constraint(proposal)
            # Ignore orders with size lower than the min ammount

        try:
            self.cleanup_shadow_orders()
            self.c_cancel_active_orders_on_max_age_limit()
            self.c_cancel_over_tolerance_orders(proposal)
            if self.c_to_create_orders(proposal):
                self.c_execute_proposal(proposal)
        except Exception as e:
            self.logger().info(e)

    cdef c_apply_bits_behind(self, object proposal, int steps):
        cdef:
            ExchangeBase market = self.exchange
        quantum_buy = market.c_get_order_price_quantum(
            self.trading_pair,
            proposal.buy.price
        )
        quantum_sell = market.c_get_order_price_quantum(
            self.trading_pair,
            proposal.sell.price
        )
        if proposal.buy.size > 0:
            proposal.buy.price -= Decimal(quantum_buy * steps)
        if proposal.sell.size > 0:
            proposal.sell.price += Decimal(quantum_sell * steps)

    cdef c_apply_moving_price_band(self, object proposal):
        price = self.get_price()
        self._moving_price_band.check_and_update_price_band(
            self.current_timestamp, price)
        if self._moving_price_band.check_price_ceiling_exceeded(price):
            proposal.buy.size = 0
        if self._moving_price_band.check_price_floor_exceeded(price):
            proposal.sell.size = 0

    def cleanup_shadow_orders(self):
        self._sb_order_tracker.check_and_cleanup_shadow_records()

    def get_price_type(self, price_type_str: str) -> PriceType:
        if price_type_str == "mid_price":
            return PriceType.MidPrice
        elif price_type_str == "best_bid":
            return PriceType.BestBid
        elif price_type_str == "best_ask":
            return PriceType.BestAsk
        elif price_type_str == "last_price":
            return PriceType.LastTrade
        elif price_type_str == 'last_own_trade_price':
            return PriceType.LastOwnTrade
        elif price_type_str == 'inventory_cost':
            return PriceType.InventoryCost
        elif price_type_str == "custom":
            return PriceType.Custom
        else:
            raise ValueError(f"Unrecognized price type string {price_type_str}.")

    def apply_ignore_over_spread(self, proposal: Proposal) -> bool:
        """apply_ignore_over_spread.
        If the calculated Order proposal spread is higher than the
        value of ignore_over_spread parameter, then the proposal is ignored.
        Works for both BUY and SELL orders.

        Args:
            proposal (Proposal): proposal
        """
        buy_spread = self.target_to_spread(proposal.buy.price)
        sell_spread = self.target_to_spread(proposal.sell.price)
        if buy_spread > self._ignore_over_spread:
            proposal.buy.size = Decimal(0)
            self.logger().info(
                'Ignoring BUY Proposal due to high spread: '
                f'ignore_over_spread = {self.ignore_over_spread * Decimal(100)}%'
            )
            return True
        if sell_spread > self._ignore_over_spread:
            proposal.sell.size = Decimal(0)
            self.logger().info(
                'Ignoring SELL Proposal due to high spread: '
                f'ignore_over_spread = {self.ignore_over_spread * Decimal(100)}%'
            )
            return True
        return False

    def has_inflight_cancel(self, order_id: str):
        return self._sb_order_tracker.has_in_flight_cancel(order_id)

    def get_price(self) -> Decimal:
        price_provider = self._asset_price_delegate or self.market_info
        if self.price_type is PriceType.LastOwnTrade:
            price = self._last_own_trade_price
        elif self.price_type is PriceType.InventoryCost:
            price = price_provider.get_price_by_type(PriceType.MidPrice)
        else:
            price = price_provider.get_price_by_type(self.price_type)

        if price.is_nan():
            price = price_provider.get_price_by_type(PriceType.MidPrice)

        return price

    def get_mid_price(self) -> Decimal:
        return self.c_get_mid_price()

