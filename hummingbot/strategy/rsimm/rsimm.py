import asyncio
from datetime import datetime
# import time
import logging
from decimal import Decimal
from statistics import mean
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import ta
from hummingbot.connector.exchange_base import ExchangeBase
from hummingbot.connector.parrot import get_campaign_summary
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.event.events import OrderType, TradeType
from hummingbot.core.rate_oracle.rate_oracle import RateOracle
from hummingbot.core.utils.estimate_fee import estimate_fee
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.market_trading_pair_tuple import \
    MarketTradingPairTuple
from hummingbot.strategy.pure_market_making.inventory_skew_calculator import \
    calculate_bid_ask_ratios_from_base_asset_ratio
from hummingbot.strategy.strategy_py_base import StrategyPyBase
from hummingbot.strategy.utils import order_age

from .historic_data import get_historic_data

hws_logger = None
NaN = float("nan")
s_decimal_zero = Decimal(0)
s_decimal_nan = Decimal("NaN")
lms_logger = None


class PriceSize:
    """
    Order price and order size.
    """
    def __init__(self, price: Decimal, size: Decimal):
        self.price: Decimal = price
        self.size: Decimal = size

    def __repr__(self):
        return f"[ p: {self.price} s: {self.size} ]"


class Proposal:
    """
    An order proposal for liquidity mining.
    market is the base quote pair like "ETH-USDT".
    buy is a buy order proposal.
    sell is a sell order proposal.
    """
    def __init__(self, market: str, buy: PriceSize, sell: PriceSize):
        self.market: str = market
        self.buy: PriceSize = buy
        self.sell: PriceSize = sell

    def __repr__(self):
        return f"{self.market} buy: {self.buy} sell: {self.sell}"

    def base(self):
        return self.market.split("-")[0]

    def quote(self):
        return self.market.split("-")[1]


class RSIMarketMaking(StrategyPyBase):
    # We use StrategyPyBase to inherit the structure. We also
    # create a logger object before adding a constructor to the class.
    @classmethod
    def logger(cls) -> HummingbotLogger:
        global hws_logger
        if hws_logger is None:
            hws_logger = logging.getLogger(__name__)
        return hws_logger

    def init_params(self,
                    exchange: ExchangeBase,
                    market_info: MarketTradingPairTuple,
                    token: str = 'USDT',
                    order_amount: Decimal = Decimal(50),
                    inventory_range_multiplier: Decimal = Decimal("1"),
                    rsi_period: int = 14,
                    rsi_overbought: int = 70,
                    rsi_oversold: int = 30,
                    rsi_interval: int = 60,
                    historic_data_from: str = '2021-10-10',
                    historic_data_to: str = datetime.today().strftime('%Y-%m-%d'),
                    historic_data_resolution: str = '1m',
                    hb_app_notification: bool = False
                    ):
        self._exchange = exchange
        self.logger().info(f'EXCHANGE: {self._exchange}')
        self._market_info = market_info
        self._token = token
        self._order_amount = order_amount
        self._inventory_range_multiplier = inventory_range_multiplier
        self._mid_prices = []
        self._rsi_period = rsi_period
        self._rsi_overbought = rsi_overbought
        self._rsi_oversold = rsi_oversold
        self._rsi_interval = rsi_interval
        self._hb_app_notification = hb_app_notification
        self._bar_period = 60
        self._historic_data_from = historic_data_from
        self._historic_data_to = historic_data_to
        self._historic_data_resolution = historic_data_resolution
        # Initialization
        self._rsi_open_position = False
        self._rsi = 100.0
        self._last_vol_reported = 0.
        self._connector_ready = False
        self._ready_to_trade = False
        self._last_vol_reported = 0.
        self._rsi_reported = 0.
        self._volatility = s_decimal_nan
        self._token_balances = {}
        self._sell_budget = s_decimal_zero
        self._buy_budget = s_decimal_zero
        self._refresh_time = 0
        self._last_rep_bar = 0
        self._bar_prices = []
        self._historic_data = pd.Series()

        self._ev_loop = asyncio.get_event_loop()
        self.add_markets([market_info.market])

        # self.fetch_historic_data()

    @property
    def market(self):
        return self._market_info.trading_pair

    @property
    def active_orders(self):
        """
        List active orders (they have been sent to the market and have not been cancelled yet)
        """
        limit_orders = self.order_tracker.active_limit_orders
        return [o[1] for o in limit_orders]

    @property
    def sell_budgets(self):
        return self._sell_budgets

    @property
    def buy_budgets(self):
        return self._buy_budget

    def cancel_active_order(self, proposal: Proposal):
        """
        Cancel any orders that have an order age greater than self._max_order_age or if orders are not within tolerance
        """
        cur_orders = [o for o in self.active_orders if o.trading_pair == proposal.market]
        # self.logger().info(f'active orders: {len(self.active_orders)}')
        # self.logger().info(f'Cur_orders: {len(cur_orders)}')
        for order in cur_orders:
            self.cancel_order(self._market_info, order.client_order_id)
            # To place new order on the next tick
            self._refresh_time = self.current_timestamp + 0.1

    def execute_orders_proposal(self, proposal: Proposal):
        """
        Execute a list of proposals if the current timestamp is less than its refresh timestamp.
        Update the refresh timestamp.
        """
        mid_price = self._market_info.get_mid_price()
        spread = s_decimal_zero
        if proposal.buy.size > 0:
            spread = abs(proposal.buy.price - mid_price) / mid_price
            self.logger().info(f"({proposal.market}) Creating a bid order {proposal.buy} value: "
                               f"{proposal.buy.size * proposal.buy.price:.2f} {proposal.quote()} spread: "
                               f"{spread:.2%}")
            self.buy_with_specific_market(
                self._market_info,
                proposal.buy.size,
                order_type=OrderType.LIMIT_MAKER,
                price=proposal.buy.price
            )
        if proposal.sell.size > 0:
            spread = abs(proposal.sell.price - mid_price) / mid_price
            self.logger().info(f"({proposal.market}) Creating an ask order at {proposal.sell} value: "
                               f"{proposal.sell.size * proposal.sell.price:.2f} {proposal.quote()} spread: "
                               f"{spread:.2%}")
            self.sell_with_specific_market(
                self._market_info,
                proposal.sell.size,
                order_type=OrderType.LIMIT_MAKER,
                price=proposal.sell.price
            )
        if proposal.buy.size > 0 or proposal.sell.size > 0:
            self._refresh_time = self.current_timestamp + self._order_refresh_time

    def is_token_a_quote_token(self):
        """
        Check if self._token is a quote token
        """
        quotes = self.all_quote_tokens()
        if len(quotes) == 1 and self._token in quotes:
            return True
        return False

    def all_base_tokens(self) -> Set[str]:
        """
        Get the base token (left-hand side) from all markets in this strategy
        """
        tokens = set()
        tokens.add(self.market.split("-")[0])
        return tokens

    def all_quote_tokens(self) -> Set[str]:
        """
        Get the quote token (right-hand side) from all markets in this strategy
        """
        tokens = set()
        tokens.add(self.market.split("-")[1])
        return tokens

    def all_tokens(self) -> Set[str]:
        """
        Return a list of all tokens involved in this strategy (base and quote)
        """
        tokens = set()
        tokens.update(self.market.split("-"))
        return tokens

    def adjusted_available_balances(self) -> Dict[str, Decimal]:
        """
        Calculates all available balances, account for amount attributed to orders and reserved balance.
        :return: a dictionary of token and its available balance
        """
        tokens = self.all_tokens()
        adjusted_bals = {t: s_decimal_zero for t in tokens}
        total_bals = {t: s_decimal_zero for t in tokens}
        total_bals.update(self._exchange.get_all_balances())
        for token in tokens:
            adjusted_bals[token] = self._exchange.get_available_balance(token)
        for order in self.active_orders:
            base, quote = order.trading_pair.split("-")
            if order.is_buy:
                adjusted_bals[quote] += order.quantity * order.price
            else:
                adjusted_bals[base] += order.quantity
        return adjusted_bals

    def apply_inventory_skew(self, proposal: Proposal):
        """
        Apply an inventory split between the quote and base asset
        """
        buy_budget = self._buy_budget
        sell_budget = self._sell_budget
        mid_price = self._market_info.get_mid_price()
        total_order_size = proposal.sell.size + proposal.buy.size
        bid_ask_ratios = calculate_bid_ask_ratios_from_base_asset_ratio(
            float(sell_budget),
            float(buy_budget),
            float(mid_price),
            float(self._target_base_pct),
            float(total_order_size * self._inventory_range_multiplier)
        )
        proposal.buy.size *= Decimal(bid_ask_ratios.bid_ratio)
        proposal.sell.size *= Decimal(bid_ask_ratios.ask_ratio)

    # After initializing the required variables, we define the tick method.
    # The tick method is the entry point for the strategy.
    def tick(self, timestamp: float):
        if not self._ready_to_trade:
            # Check if there are restored orders, they should be canceled before strategy starts.
            self._ready_to_trade = self._exchange.ready and len(self._exchange.limit_orders) == 0
            if not self._exchange.ready:
                self.logger().warning(f"{self._exchange.name} is not ready. Please wait...")
                return
            else:
                self.logger().info(f"{self._exchange.name} is ready. Trading started.")
                self.create_budget_allocation()
        self.update_mid_price()
        self.update_bar_price()
        self.update_rsi()
        # proposal = self.decision()

        # self._token_balances = self.adjusted_available_balances()
        # self.apply_budget_constraint(proposal)
        # self.execute_orders_proposal(proposal)
        self._last_timestamp = timestamp

    def decision(self) -> Proposal:
        proposal = Proposal(
            self.market,
            PriceSize(0, 0),
            PriceSize(0, 0)
        )

        if self._rsi > self._rsi_overbought:
            if self._rsi_open_position:
                self.logger().info("Overbought! SELL!")
                self.sell_with_specific_market(
                    self._market_info,
                    self._order_amount,
                    order_type=OrderType.MARKET,
                    price=self._mid_prices[-1]
                )
                proposal = Proposal(
                    self.market,
                    PriceSize(0, 0),
                    PriceSize(self._mid_prices[-1], self._order_amount)
                )
                self._rsi_open_position = False
            else:
                pass
        elif self._rsi < self._rsi_oversold:
            if self._rsi_open_position:
                pass
            else:
                self.logger().info("Oversold! BUY!")
                self.buy_with_specific_market(
                    self._market_info,
                    self._order_amount,
                    order_type=OrderType.MARKET,
                    price=self._mid_prices[-1]
                )
                proposal = Proposal(
                    self.market,
                    PriceSize(self._mid_prices[-1], self._order_amount),
                    PriceSize(0, 0)
                )
                self._rsi_open_position = True
        return proposal

    async def active_orders_df(self) -> pd.DataFrame:
        """
        Return the active orders in a DataFrame.
        """
        size_q_col = f"Amt({self._token})" if self.is_token_a_quote_token() else "Amt(Quote)"
        columns = ["Market", "Side", "Price", "Spread", "Amount", size_q_col, "Age"]
        data = []
        for order in self._active_orders:
            mid_price = self._market_info.get_mid_price()
            spread = 0 if mid_price == 0 else abs(order.price - mid_price) / mid_price
            size_q = order.quantity * mid_price
            age = order_age(order)
            # // indicates order is a paper order so 'n/a'. For real orders, calculate age.
            age_txt = "n/a" if age <= 0. else pd.Timestamp(age, unit='s').strftime('%H:%M:%S')
            data.append([
                order.trading_pair,
                "buy" if order.is_buy else "sell",
                float(order.price),
                f"{spread:.2%}",
                float(order.quantity),
                float(size_q),
                age_txt
            ])
        df = pd.DataFrame(data=data, columns=columns)
        df.sort_values(by=["Market", "Side"], inplace=True)
        return df

    def budget_status_df(self) -> pd.DataFrame:
        """
        Return the trader's budget in a DataFrame
        """
        data = []
        columns = ["Market", f"Budget({self._token})", "Base bal", "Quote bal", "Base/Quote"]
        market_info = self._market_info
        market = self.market
        mid_price = market_info.get_mid_price()
        base_bal = self._sell_budget
        quote_bal = self._buy_budget
        total_bal_in_quote = (base_bal * mid_price) + quote_bal
        total_bal_in_token = total_bal_in_quote
        if not self.is_token_a_quote_token():
            total_bal_in_token = base_bal + (quote_bal / mid_price)
        base_pct = (base_bal * mid_price) / total_bal_in_quote if total_bal_in_quote > 0 else s_decimal_zero
        quote_pct = quote_bal / total_bal_in_quote if total_bal_in_quote > 0 else s_decimal_zero
        data.append([
            market,
            float(total_bal_in_token),
            float(base_bal),
            float(quote_bal),
            f"{base_pct:.0%} / {quote_pct:.0%}"
        ])
        df = pd.DataFrame(data=data, columns=columns).replace(np.nan, '', regex=True)
        df.sort_values(by=["Market"], inplace=True)
        return df

    async def market_status_df(self) -> pd.DataFrame:
        """
        Return the market status (prices, volatility) in a DataFrame
        """
        data = []
        columns = ["Market", "Mid price", "Best bid", "Best ask", "Volatility"]
        market_info = self._market_info
        market = self.market
        mid_price = market_info.get_mid_price()
        best_bid = self._exchange.get_price(market, False)
        best_ask = self._exchange.get_price(market, True)
        best_bid_pct = abs(best_bid - mid_price) / mid_price
        best_ask_pct = (best_ask - mid_price) / mid_price
        data.append([
            market,
            float(mid_price),
            f"{best_bid_pct:.2%}",
            f"{best_ask_pct:.2%}",
            "" if self._volatility.is_nan() else f"{self._volatility:.2%}",
        ])
        df = pd.DataFrame(data=data, columns=columns).replace(np.nan, '', regex=True)
        df.sort_values(by=["Market"], inplace=True)
        return df

    async def miner_status_df(self) -> pd.DataFrame:
        """
        Return the miner status (payouts, rewards, liquidity, etc.) in a DataFrame
        """
        data = []
        g_sym = RateOracle.global_token_symbol
        columns = ["Market", "Payout", "Reward/wk", "Liquidity", "Yield/yr", "Max spread"]
        campaigns = await get_campaign_summary(self._exchange.display_name,
                                               self._market_info.market.name)
        for market, campaign in campaigns.items():
            reward = await RateOracle.global_value(campaign.payout_asset, campaign.reward_per_wk)
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

    async def format_status(self) -> str:
        """
        Return the budget, market, miner and order statuses.
        """
        if not self._ready_to_trade:
            return "Market connectors are not ready."
        lines = []
        warning_lines = []
        warning_lines.extend(self.network_warning(self._market_info))

        budget_df = self.budget_status_df()
        lines.extend(["", "  Budget:"] + ["    " + line for line in budget_df.to_string(index=False).split("\n")])

        market_df = self.market_status_df()
        lines.extend(["", "  Markets:"] + ["    " + line for line in market_df.to_string(index=False).split("\n")])

        miner_df = await self.miner_status_df()
        if not miner_df.empty:
            lines.extend(["", "  Miner:"] + ["    " + line for line in miner_df.to_string(index=False).split("\n")])

        # See if there are any open orders.
        if len(self.active_orders) > 0:
            df = await self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        else:
            lines.extend(["", "  No active maker orders."])

        warning_lines.extend(self.balance_warning(self._market_info))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)

    def did_complete_buy_order(self, order_completed_event):
        self.logger().info(f"LimitMaker Order {order_completed_event.order_id} has been executed")
        self.logger().info(order_completed_event)

    def did_fill_order(self, event):
        """
        Check if order has been completed, log it, notify the hummingbot application, and update budgets.
        """
        order_id = event.order_id
        market_info = self.order_tracker.get_shadow_market_pair_from_order_id(order_id)
        if market_info is not None:
            if event.trade_type is TradeType.BUY:
                msg = f"({market_info.trading_pair}) Maker BUY order (price: {event.price}) of {event.amount} " \
                      f"{market_info.base_asset} is filled."
                self.log_with_clock(logging.INFO, msg)
                self.notify_hb_app_with_timestamp(msg)
                self._buy_budget -= (event.amount * event.price)
                self._sell_budget += event.amount
            else:
                msg = f"({market_info.trading_pair}) Maker SELL order (price: {event.price}) of {event.amount} " \
                      f"{market_info.base_asset} is filled."
                self.log_with_clock(logging.INFO, msg)
                self.notify_hb_app_with_timestamp(msg)
                self._sell_budget -= event.amount
                self._buy_budget += (event.amount * event.price)

    def start(self, clock: Clock, timestamp: float):
        restored_orders = self._exchange.limit_orders
        for order in restored_orders:
            self._exchange.cancel(order.trading_pair, order.client_order_id)

    def total_port_value_in_token(self) -> Decimal:
        """
        Total portfolio value in self._token amount
        """
        market_info = self._market_info
        market = self.market
        all_bals = self.adjusted_available_balances()
        port_value = all_bals.get(self._token, s_decimal_zero)
        base, quote = market.split("-")
        if self.is_token_a_quote_token():
            port_value += all_bals[base] * market_info.get_mid_price()
        else:
            port_value += all_bals[quote] / market_info.get_mid_price()
        return port_value

    def create_budget_allocation(self):
        """
        Create buy and sell budgets for every market
        """
        market_info = self._market_info
        market = self.market
        portfolio_value = self.total_port_value_in_token()
        market_portion = portfolio_value
        balances = self.adjusted_available_balances()
        base, quote = market.split("-")
        if self.is_token_a_quote_token():
            self._sell_budget = balances[base]
            buy_budget = market_portion - (balances[base] * market_info.get_mid_price())
            if buy_budget > s_decimal_zero:
                self._buy_budget = buy_budget
        else:
            self._buy_budget = balances[quote]
            sell_budget = market_portion - (balances[quote] / market_info.get_mid_price())
            if sell_budget > s_decimal_zero:
                self._sell_budget = sell_budget

    def base_order_size(self, trading_pair: str, price: Decimal = s_decimal_zero):
        base, quote = trading_pair.split("-")
        if self._token == base:
            return self._order_amount
        if price == s_decimal_zero:
            price = self._market_info.get_mid_price()
        return self._order_amount / price

    def apply_budget_constraint(self, proposal: Proposal):
        balances = self._token_balances.copy()

        if balances[proposal.base()] < proposal.sell.size:
            proposal.sell.size = balances[proposal.base()]
        proposal.sell.size = self._exchange.quantize_order_amount(proposal.market, proposal.sell.size)
        balances[proposal.base()] -= proposal.sell.size

        quote_size = proposal.buy.size * proposal.buy.price
        quote_size = balances[proposal.quote()] if balances[proposal.quote()] < quote_size else quote_size
        buy_fee = estimate_fee(self._exchange.name, True)
        buy_size = quote_size / (proposal.buy.price * (Decimal("1") + buy_fee.percent))
        proposal.buy.size = self._exchange.quantize_order_amount(proposal.market, buy_size)
        balances[proposal.quote()] -= quote_size

    def is_within_tolerance(self, cur_orders: List[LimitOrder], proposal: Proposal):
        """
        False if there are no buys or sells or if the difference between the proposed price and current price is less
        than the tolerance. The tolerance value is strict max, cannot be equal.
        """
        cur_buy = [o for o in cur_orders if o.is_buy]
        cur_sell = [o for o in cur_orders if not o.is_buy]
        if (cur_buy and proposal.buy.size <= 0) or (cur_sell and proposal.sell.size <= 0):
            return False
        if cur_buy and \
                abs(proposal.buy.price - cur_buy[0].price) / cur_buy[0].price > self._order_refresh_tolerance_pct:
            return False
        if cur_sell and \
                abs(proposal.sell.price - cur_sell[0].price) / cur_sell[0].price > self._order_refresh_tolerance_pct:
            return False
        return True

    def notify_hb_app(self, msg: str):
        """
        Send a message to the hummingbot application
        """
        if self._hb_app_notification:
            super().notify_hb_app(msg)

    def update_mid_price(self):
        """
        Update mid-price data from the market
        """
        mid_price = self._market_info.get_mid_price()
        # self.logger().info(f'Market mid-price: {mid_price}')
        self._mid_prices.append(mid_price)
        # To avoid memory leak, we store only the last part of the list needed
        # for volatility/rsi calculation
        max_len = self._avg_volatility_period * self._volatility_interval
        if self._rsi_interval > max_len:
            max_len = self._rsi_interval
        if self._bar_period > max_len:
            max_len = self._bar_period
        self._mid_prices = self._mid_prices[-1 * max_len:]

    def update_bar_price(self):
        """
        Update mean bar price data from the market
        """
        if self.current_timestamp - self._last_rep_bar > self._bar_period:
            bar_price = float(mean(self._mid_prices[-1 * self._bar_period:]))
            now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            self._bar_prices.append((now, bar_price))
            self._last_rep_bar = self.current_timestamp
            self.logger().info(
                f'Last Bar ({self._bar_period} seconds) mid-price: {bar_price}'
            )

    def fetch_historic_data(self):
        pair = self.market.replace('-', '/')
        self.logger().info('Please wait while downloading historic data...')
        data = get_historic_data(pair,
                                 self._historic_data_from,
                                 self._historic_data_to,
                                 resolution=self._historic_data_resolution)
        self.logger().info('Finished downloading historic data.')
        self.logger().info(
            f'Historic data [{self._historic_data_from}, ' +
            f'{self._historic_data_to}] for <{pair}>: ' +
            f'Len={data.size}, Resolution={self._historic_data_resolution}'
        )
        self._historic_data = data['close']
        print(self._historic_data)
        print(type(self._historic_data))

    def update_rsi(self):
        data = self._historic_data.append(pd.Series([k[1] for k in self._bar_prices]))
        if data.size >= self._rsi_period:
            if self._rsi_reported < self.current_timestamp - self._rsi_interval:
                rsi = self.calculate_rsi(data)
                rsi2 = self.calc_rsi(data)
                if rsi is None or rsi2 is None:
                    return
                last_rsi = rsi.iloc[-1]
                last_rsi2 = rsi2.iloc[-1]

                self._rsi_reported = self.current_timestamp
                self.logger().info(f"RSI: {last_rsi}")
                self.logger().info(f"RSI2: {last_rsi2}")
                rsi_mean = mean(rsi[- self._rsi_interval:])
                rsi2_mean = mean(rsi2[- self._rsi_interval:])
                self.logger().info(f"RSI Mean: {rsi_mean}")
                self.logger().info(f"RSI2 Mean: {rsi2_mean}")
        else:
            # self.logger().info('Not enough samples to calculate RSI')
            return

    def calculate_rsi(self, prices):
        rsi = ta.momentum.rsi(prices, self._rsi_period, True)
        return rsi

    def calc_rsi(self, prices):
        period = self._rsi_period
        delta = prices.diff().dropna()
        ups = delta * 0
        downs = ups.copy()
        ups[delta > 0] = delta[delta > 0]
        downs[delta < 0] = -delta[delta < 0]
        # first value is sum of avg gains
        ups[ups.index[period - 1]] = np.mean(ups[:period])
        ups = ups.drop(ups.index[:(period - 1)])
        # first value is sum of avg losses
        downs[downs.index[period - 1]] = np.mean(downs[:period])
        downs = downs.drop(downs.index[:(period - 1)])
        rs = ups.ewm(com=period - 1,
                     min_periods=0,
                     adjust=False,
                     ignore_na=False).mean() / \
            downs.ewm(com=period - 1,
                      min_periods=0,
                      adjust=False,
                      ignore_na=False).mean()
        rsi = 100 - 100 / (1 + rs)
        return rsi
