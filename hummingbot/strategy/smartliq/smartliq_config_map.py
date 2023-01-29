"""
The configuration parameters for a user made smartliq strategy.
"""

from decimal import Decimal
from typing import Optional

from hummingbot.client.config.config_validators import (
    validate_bool,
    validate_connector,
    validate_decimal,
    validate_exchange,
    validate_int,
    validate_market_trading_pair,
)
from hummingbot.client.config.config_var import ConfigVar
from hummingbot.client.settings import AllConnectorSettings, required_exchanges


def validate_price_source(value: str) -> Optional[str]:
    if value not in {"current_market", "external_market", "custom_api"}:
        return "Invalid price source type."


def on_validate_price_source(value: str):
    if value != "external_market":
        smartliq_config_map["price_source_exchange"].value = None
        smartliq_config_map["price_source_market"].value = None
    if value != "custom_api":
        smartliq_config_map["price_source_custom_api"].value = None
    else:
        smartliq_config_map["price_type"].value = "custom"


def price_source_market_prompt() -> str:
    external_market = smartliq_config_map.get("price_source_exchange").value
    return f'Enter the token trading pair on {external_market} --> '


def validate_price_source_exchange(value: str) -> Optional[str]:
    if value == smartliq_config_map.get("exchange").value:
        return "Price source exchange cannot be the same as maker exchange."
    return validate_connector(value)


def on_validated_price_source_exchange(value: str):
    if value is None:
        smartliq_config_map.get("price_source_market").value = None


def validate_price_source_market(value: str) -> Optional[str]:
    market = smartliq_config_map.get("price_source_exchange").value
    return validate_market_trading_pair(market, value)


def validate_price_floor_ceiling(value: str) -> Optional[str]:
    try:
        decimal_value = Decimal(value)
    except Exception:
        return f"{value} is not in decimal format."
    if not (decimal_value == Decimal("-1") or decimal_value > Decimal("0")):
        return "Value must be more than 0 or -1 to disable this feature."


def validate_price_type(value: str) -> Optional[str]:
    error = None
    price_source = smartliq_config_map.get("price_source").value
    if price_source != "custom_api":
        valid_values = {"mid_price",
                        "last_price",
                        "last_own_trade_price",
                        "best_bid",
                        "best_ask",
                        "inventory_cost",
                        }
        if value not in valid_values:
            error = "Invalid price type."
    elif value != "custom":
        error = "Invalid price type."
    return error


def on_validated_price_type(value: str):
    if value == 'inventory_cost':
        smartliq_config_map["inventory_price"].value = None


def exchange_on_validated(value: str) -> None:
    required_exchanges.add(value)


def validate_vol_algo(value: str) -> Optional[str]:
    valid = True if value in ('ivol', 'hvol', 'ewm-var', 'ewm-vol') else False
    if not valid:
        return f'Algorithm <{value}> does not exist!'


def validate_exchange_trading_pair(value: str) -> Optional[str]:
    exchange = smartliq_config_map.get("exchange").value
    return validate_market_trading_pair(exchange, value)


def maker_trading_pair_prompt():
    exchange = smartliq_config_map.get("exchange").value
    example = AllConnectorSettings.get_example_pairs().get(exchange)
    return "Enter the token trading pair you would like to trade on %s%s --> " \
           % (exchange, f" (e.g. {example})" if example else "")


def token_validate(value: str) -> Optional[str]:
    value = value.upper()
    market = smartliq_config_map["market"].value
    tokens = set()
    # Tokens in markets already validated in market_validate()
    for token in market.strip().upper().split("-"):
        tokens.add(token.strip())
    if value not in tokens:
        return f"Invalid token. {value} is not one of {','.join(sorted(tokens))}"


def order_amount_prompt() -> str:
    token = smartliq_config_map["token"].value
    return f"What is the size of each order (in {token} amount)? --> "


smartliq_config_map = {
    "strategy":
        ConfigVar(
            key="strategy",
            prompt="",
            default="smartliq"
        ),
    "exchange":
        ConfigVar(
            key="exchange",
            prompt="Enter the spot connector to use for smarliq mining --> ",
            validator=validate_exchange,
            on_validated=exchange_on_validated,
            prompt_on_new=True
        ),
    "market":
        ConfigVar(
            key="market",
            prompt=maker_trading_pair_prompt,
            type_str="str",
            validator=validate_exchange_trading_pair,
            prompt_on_new=True
        ),
    "token":
        ConfigVar(
            key="token",
            prompt="What asset (base or quote) do you want to use "
                "to provide liquidity? --> ",
            type_str="str",
            validator=token_validate,
            prompt_on_new=True
        ),
    "order_amount":
        ConfigVar(
            key="order_amount",
            prompt=order_amount_prompt,
            type_str="decimal",
            validator=lambda v: validate_decimal(v, min_value=Decimal("0"), inclusive=False),
            prompt_on_new=True
        ),
    "inventory_skew_enabled":
        ConfigVar(
            key="inventory_skew_enabled",
            prompt="Would you like to enable inventory skew? (Yes/No) --> ",
            type_str="bool",
            default=False,
            validator=validate_bool,
            prompt_on_new=False
        ),
    "inventory_target_base_pct":
        ConfigVar(
            key="inventory_target_base_pct",
            prompt="For each pair, what is your target base asset "
                   "percentage? (Enter 20 to indicate 20%) --> ",
            type_str="decimal",
            default=Decimal("1"),
            validator=lambda v: validate_decimal(v, 0, 100, inclusive=True),
            prompt_on_new=False
        ),
    "order_refresh_time":
        ConfigVar(
            key="order_refresh_time",
            prompt="How often do you want to cancel and replace bids and asks "
                   "(in seconds)? --> ",
            type_str="float",
            default=1,
            validator=lambda v: validate_decimal(v, 0, inclusive=False),
            prompt_on_new=True
        ),
    "order_refresh_tolerance_pct":
        ConfigVar(
            key="order_refresh_tolerance_pct",
            prompt="Enter the percent change in price needed to refresh orders at each cycle "
                "(Enter 1 to indicate 1%) --> ",
            type_str="decimal",
            default=Decimal("0"),
            validator=lambda v: validate_decimal(v, 0, 100, inclusive=True),
            prompt_on_new=False
        ),
    "inventory_range_multiplier":
        ConfigVar(
            key="inventory_range_multiplier",
            prompt="What is your tolerable range of inventory around the target, "
                "expressed in multiples of your total order size? ",
            type_str="decimal",
            validator=lambda v: validate_decimal(v, min_value=0, inclusive=False),
            default=Decimal("1"),
            prompt_on_new=False
        ),
    "volatility_price_samples":
        ConfigVar(
            key="volatility_price_samples",
            prompt="Number of mid price data samples to use to calculate "
                "market volatility on each cycle? --> ",
            type_str="int",
            validator=lambda v: validate_int(v, min_value=1, inclusive=False),
            default=2,
            prompt_on_new=False
        ),
    "volatility_interval":
        ConfigVar(
            key="volatility_interval",
            prompt="The interval, in second, in which to pick mid price "
                " data from to calculate market volatility --> ",
            type_str="int",
            validator=lambda v: validate_int(v, min_value=0, inclusive=False),
            default=1,
            prompt_on_new=False
        ),
    "avg_volatility_samples":
        ConfigVar(
            key="avg_volatility_samples",
            prompt="How many volatility samples to use to"
                " to calculate average market volatility? --> ",
            type_str="int",
            validator=lambda v: validate_int(v, min_value=0, inclusive=False),
            default=1,
            prompt_on_new=False
        ),
    "volatility_to_spread_multiplier":
        ConfigVar(
            key="volatility_to_spread_multiplier",
            prompt="Enter a multiplier used to convert average volatility to spread "
                   "(enter 1 for 1 to 1 conversion) --> ",
            type_str="decimal",
            validator=lambda v: validate_decimal(v, min_value=Decimal('0'), inclusive=False),
            default=Decimal("10"),
            prompt_on_new=False
        ),
    "volatility_algorithm":
        ConfigVar(
            key="volatility_algorithm",
            prompt="Select the Volatility Algorithm to use for smarliq mining"
                   " (ivol, hvol, ewm-var, ewm-vol) --> ",
            type_str="str",
            default="ivol",
            validator=lambda v: validate_vol_algo(v),
            prompt_on_new=False
        ),
    "max_spread":
        ConfigVar(
            key="max_spread",
            prompt="What is the maximum spread? (Enter 1 to indicate "
                   "1% or -1 to ignore this setting) --> ",
            type_str="decimal",
            validator=lambda v: validate_decimal(v),
            default=Decimal("-1")
        ),
    "max_order_age":
        ConfigVar(
            key="max_order_age",
            prompt="What is the maximum life time of your orders "
                   "(in seconds)? --> ",
            type_str="float",
            validator=lambda v: validate_decimal(v, min_value=0, inclusive=False),
            default=1800
        ),
    "bid_position":
        ConfigVar(
            key="bid_position",
            prompt="Target Bid position on the order book (Default=3) --> ",
            type_str="int",
            validator=lambda v: validate_int(v, inclusive=False),
            default=3,
            prompt_on_new=True
        ),
    "ask_position":
        ConfigVar(
            key="ask_position",
            prompt="Target Ask position on the order book(Default=3) --> ",
            type_str="int",
            validator=lambda v: validate_int(v, inclusive=False),
            default=3,
            prompt_on_new=True
        ),
    "buy_volume_in_front":
        ConfigVar(
            key="buy_volume_in_front",
            prompt="Buy (bid) volume in the order book in front of order (Default=-1) --> ",
            type_str="decimal",
            validator=lambda v: validate_decimal(v, inclusive=False),
            default=Decimal("-1")
        ),
    "sell_volume_in_front":
        ConfigVar(
            key="sell_volume_in_front",
            prompt="Sell (ask) volume in the order book in front of order (Default=-1) --> ",
            type_str="decimal",
            validator=lambda v: validate_decimal(v, inclusive=False),
            default=Decimal("-1")
        ),
    "ignore_over_spread":
        ConfigVar(
            key="ignore_over_spread",
            prompt="Ignore Order Proposals with spread higher than this value "
                   "(%)? --> ",
            type_str="decimal",
            default=Decimal("1"),
            validator=lambda v: validate_decimal(v, min_value=Decimal("0"), inclusive=False),
        ),
    "filled_order_delay":
        ConfigVar(
            key="filled_order_delay",
            prompt="Delay after a filled order --> ",
            type_str="float",
            default=10.0,
            validator=lambda v: validate_decimal(v, 0, inclusive=True),
        ),
    "bits_behind":
        ConfigVar(
            key="bits_behind",
            prompt="Set steps for the bits-behind feature --> ",
            type_str="int",
            validator=lambda v: validate_int(v, inclusive=True),
            default=1
        ),
    "price_source":
        ConfigVar(
            key="price_source",
            prompt="Which price source to use? (current_market/external_market/custom_api) --> ",
            type_str="str",
            default="current_market",
            validator=validate_price_source,
            on_validated=on_validate_price_source
        ),
    "price_type":
        ConfigVar(
            key="price_type",
            prompt="Which price type to use? ("
                   "mid_price/last_price/last_own_trade_price/best_bid/best_ask/inventory_cost) --> ",
            type_str="str",
            required_if=lambda: smartliq_config_map.get("price_source").value != "custom_api",
            default="mid_price",
            on_validated=on_validated_price_type,
            validator=validate_price_type
        ),
    "price_source_exchange":
        ConfigVar(
            key="price_source_exchange",
            prompt="Enter external price source exchange name --> ",
            required_if=lambda: smartliq_config_map.get("price_source").value == "external_market",
            type_str="str",
            validator=validate_price_source_exchange,
            on_validated=on_validated_price_source_exchange
        ),
    "price_source_market":
        ConfigVar(
            key="price_source_market",
            prompt=price_source_market_prompt,
            required_if=lambda: smartliq_config_map.get("price_source").value == "external_market",
            type_str="str",
            validator=validate_price_source_market
        ),
    "price_source_custom_api":
        ConfigVar(
            key="price_source_custom_api",
            prompt="Enter pricing API URL --> ",
            required_if=lambda: smartliq_config_map.get("price_source").value == "custom_api",
            type_str="str"
        ),
    "custom_api_update_interval":
        ConfigVar(
            key="custom_api_update_interval",
            prompt="Enter custom API update interval in second (default: 5.0, min: 0.5) --> ",
            required_if=lambda: False,
            default=float(5),
            type_str="float",
            validator=lambda v: validate_decimal(v, Decimal("0.5"))
        ),
    "moving_price_band_enabled":
        ConfigVar(
            key="moving_price_band_enabled",
            prompt="Would you like to enable moving price floor and ceiling? (Yes/No) --> ",
            type_str="bool",
            default=False,
            validator=validate_bool
        ),
    "price_ceiling_pct":
        ConfigVar(
            key="price_ceiling_pct",
            prompt="Enter a percentage to the current price that sets the price ceiling. Above this price, only sell orders will be placed --> ",
            type_str="decimal",
            default=Decimal("1"),
            required_if=lambda: smartliq_config_map.get("moving_price_band_enabled").value,
            validator=validate_decimal
        ),
    "price_floor_pct":
        ConfigVar(
            key="price_floor_pct",
            prompt="Enter a percentage to the current price that sets the price floor. Below this price, only buy orders will be placed --> ",
            type_str="decimal",
            default=Decimal("-1"),
            required_if=lambda: smartliq_config_map.get("moving_price_band_enabled").value,
            validator=validate_decimal
        ),
    "price_band_refresh_time":
        ConfigVar(
            key="price_band_refresh_time",
            prompt="After this amount of time (in seconds), the price bands are reset based on the current price --> ",
            type_str="float",
            default=86400,
            required_if=lambda: smartliq_config_map.get("moving_price_band_enabled").value,
            validator=validate_decimal
        ),
}
