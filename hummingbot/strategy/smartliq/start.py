from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.connector.exchange.paper_trade import create_paper_trade_market
from hummingbot.connector.exchange_base import ExchangeBase
from hummingbot.strategy.api_asset_price_delegate import APIAssetPriceDelegate
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.order_book_asset_price_delegate import OrderBookAssetPriceDelegate
from hummingbot.strategy.pure_market_making import InventoryCostPriceDelegate
from hummingbot.strategy.pure_market_making.moving_price_band import MovingPriceBand
from hummingbot.strategy.smartliq.smartliq import SmartLiquidityStrategy
from hummingbot.strategy.smartliq.smartliq_config_map import smartliq_config_map as c_map


def start(self):
    exchange = c_map.get("exchange").value.lower()
    el_market = c_map.get("market").value.strip().upper()
    token = c_map.get("token").value.upper()
    quote_market = el_market if el_market.split('-')[1] == token else None
    base_market = el_market if el_market.split('-')[0] == token else None
    market = quote_market if quote_market else base_market
    order_amount = c_map.get("order_amount").value
    buy_spread = c_map.get("buy_spread").value
    sell_spread = c_map.get("sell_spread").value
    inventory_skew_enabled = c_map.get("inventory_skew_enabled").value
    inventory_target_base_pct = 0 if c_map.get("inventory_target_base_pct").value is None else \
        c_map.get("inventory_target_base_pct").value
    order_refresh_time = c_map.get("order_refresh_time").value
    order_refresh_tolerance_pct = c_map.get("order_refresh_tolerance_pct").value
    inventory_range_multiplier = c_map.get("inventory_range_multiplier").value
    volatility_price_samples = c_map.get("volatility_price_samples").value
    volatility_interval = c_map.get("volatility_interval").value
    avg_volatility_samples = c_map.get("avg_volatility_samples").value
    volatility_to_spread_multiplier = c_map.get("volatility_to_spread_multiplier").value
    max_spread = c_map.get("max_spread").value
    max_order_age = c_map.get("max_order_age").value
    order_book_depth = c_map.get("order_book_depth").value
    order_book_log_interval = c_map.get("order_book_log_interval").value
    order_book_buy_position = c_map.get("order_book_buy_position").value
    order_book_sell_position = c_map.get("order_book_sell_position").value
    order_book_buy_volume_in_front = c_map.get("order_book_buy_volume_in_front").value
    order_book_sell_volume_in_front = c_map.get("order_book_sell_volume_in_front").value
    ping_pong_enabled = c_map.get("ping_pong_enabled").value
    ping_pong_initial_buy = c_map.get("ping_pong_initial_buy").value
    restart_every_seconds = c_map.get("restart_every_seconds").value
    volatility_algorithm = c_map.get("volatility_algorithm").value
    ignore_over_spread = c_map.get("ignore_over_spread").value
    min_order_usdt = c_map.get("min_order_usdt").value
    filled_order_delay = c_map.get("filled_order_delay").value
    bits_behind = c_map.get("bits_behind").value

    price_source = c_map.get("price_source").value
    price_type = c_map.get("price_type").value
    price_source_exchange = c_map.get("price_source_exchange").value
    price_source_market = c_map.get("price_source_market").value
    price_source_custom_api = c_map.get("price_source_custom_api").value
    custom_api_update_interval = c_map.get("custom_api_update_interval").value

    moving_price_band = MovingPriceBand(
        enabled=c_map.get("moving_price_band_enabled").value,
        price_floor_pct=c_map.get("price_floor_pct").value,
        price_ceiling_pct=c_map.get("price_ceiling_pct").value,
        price_band_refresh_time=c_map.get("price_band_refresh_time").value
    )

    asset_price_delegate = None
    if price_source == "external_market":
        asset_trading_pair: str = price_source_market
        ext_market = create_paper_trade_market(price_source_exchange, [asset_trading_pair])
        self.markets[price_source_exchange]: ExchangeBase = ext_market
        asset_price_delegate = OrderBookAssetPriceDelegate(ext_market, asset_trading_pair)
    elif price_source == "custom_api":
        ext_market = create_paper_trade_market(exchange, [el_market])
        asset_price_delegate = APIAssetPriceDelegate(ext_market, price_source_custom_api,
                                                     custom_api_update_interval)

    inventory_cost_price_delegate = None
    if price_type == "inventory_cost":
        db = HummingbotApplication.main_application().trade_fill_db
        inventory_cost_price_delegate = InventoryCostPriceDelegate(db, el_market)

    self._initialize_markets([(exchange, [market])])
    exchange = self.markets[exchange]
    base, quote = market.split("-")
    market_info = MarketTradingPairTuple(exchange, market, base, quote)
    self.strategy = SmartLiquidityStrategy()
    self.strategy.init_params(
        market_info=market_info,
        token=token,
        order_amount=order_amount,
        buy_spread=buy_spread,
        sell_spread=sell_spread,
        inventory_skew_enabled=inventory_skew_enabled,
        inventory_target_base_pct=inventory_target_base_pct,
        order_refresh_time=order_refresh_time,
        order_refresh_tolerance_pct=order_refresh_tolerance_pct,
        inventory_range_multiplier=inventory_range_multiplier,
        volatility_price_samples=volatility_price_samples,
        volatility_interval=volatility_interval,
        avg_volatility_samples=avg_volatility_samples,
        volatility_to_spread_multiplier=volatility_to_spread_multiplier,
        volatility_algorithm=volatility_algorithm,
        max_spread=max_spread,
        max_order_age=max_order_age,
        order_book_depth=order_book_depth,
        order_book_log_interval=order_book_log_interval,
        order_book_buy_position=order_book_buy_position,
        order_book_sell_position=order_book_sell_position,
        order_book_buy_volume_in_front=order_book_buy_volume_in_front,
        order_book_sell_volume_in_front=order_book_sell_volume_in_front,
        ping_pong_enabled=ping_pong_enabled,
        ping_pong_initial_buy=ping_pong_initial_buy,
        restart_every_seconds=restart_every_seconds,
        ignore_over_spread=ignore_over_spread,
        min_order_usdt=min_order_usdt,
        filled_order_delay=filled_order_delay,
        bits_behind=bits_behind,
        asset_price_delegate=asset_price_delegate,
        inventory_cost_price_delegate=inventory_cost_price_delegate,
        moving_price_band=moving_price_band,
        hb_app_notification=True,
    )
