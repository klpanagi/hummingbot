from decimal import Decimal
from typing import Optional

from hummingbot.client.config.config_var import ConfigVar
from hummingbot.client.config.config_validators import (
    validate_exchange,
    validate_decimal,
    validate_int,
    validate_bool
)

from hummingbot.client.settings import (
    required_exchanges,
)


def exchange_on_validated(value: str) -> None:
    required_exchanges.append(value)

# Returns a market prompt that incorporates the connector value set by the user
def market_prompt() -> str:
    exchange = smartliq_config_map.get("exchange").value
    return f'Enter the token trading pair on {exchange} >>> '


def order_size_prompt() -> str:
    token = smartliq_config_map["token"].value
    return f"What is the size of each order (in {token} amount)? >>> "


def token_validate(value: str) -> Optional[str]:
    value = value.upper()
    market = smartliq_config_map["market"].value
    tokens = market.strip().upper().split("-")
    tokens = [token.strip() for token in tokens]
    if value not in tokens:
        return f"Invalid token. {value} is not one of {','.join(sorted(tokens))}"



# List of parameters defined by the strategy
smartliq_config_map ={
    "strategy":
        ConfigVar(key="strategy",
                  prompt="",
                  default="smartliq",
        ),
    "exchange":
        ConfigVar(key="exchange",
                  prompt="Enter the spot connector to use for liquidity mining >>> ",
                  validator=validate_exchange,
                  on_validated=exchange_on_validated,
                  prompt_on_new=True),
    "market":
        ConfigVar(key="market",
                  prompt=market_prompt,
                  prompt_on_new=True,
        ),
    "token":
        ConfigVar(key="token",
                  prompt="What asset (base or quote) do you want to use? >>> ",
                  type_str="str",
                  validator=token_validate,
                  default='USDT',
                  prompt_on_new=True),
    "order_amount":
        ConfigVar(key="order_amount",
                  prompt=order_size_prompt,
                  type_str="decimal",
                  prompt_on_new=True
    ),
    "spread":
        ConfigVar(key="spread",
                  prompt="How far away from the mid price do you want to place bid and ask orders? (Enter 1 to indicate 1%) >>> ",
                  type_str="decimal",
                  validator=lambda v: validate_decimal(v, 0, 100, inclusive=False),
                  prompt_on_new=True),
    "order_refresh_time":
        ConfigVar(key="order_refresh_time",
                  prompt="How often do you want to cancel and replace bids and asks (in seconds)? >>> ",
                  type_str="float",
                  validator=lambda v: validate_decimal(v, 0, inclusive=False),
                  default=10.),
    "order_refresh_tolerance_pct":
        ConfigVar(key="order_refresh_tolerance_pct",
                  prompt="Enter the percent change in price needed to refresh orders at each cycle (Enter 1 to indicate 1%) >>> ",
                  type_str="decimal",
                  default=Decimal("0.2"),
                  validator=lambda v: validate_decimal(v, -10, 10, inclusive=True)),
    "volatility_interval":
        ConfigVar(key="volatility_interval",
                  prompt="What is an interval, in second, in which to pick historical mid price data from to calculate market volatility? >>> ",
                  type_str="int",
                  default=10),
    "avg_volatility_period":
        ConfigVar(key="avg_volatility_period",
                  prompt="How many interval samples to use to calculate average market volatility? >>> ",
                  type_str="int",
                  default=2),
    "volatility_to_spread_multiplier":
        ConfigVar(key="volatility_to_spread_multiplier",
                  prompt="Enter a multiplier used to convert average volatility to spread (enter 1 for 1 to 1 conversion) >>> ",
                  type_str="decimal",
                  validator=lambda v: validate_decimal(v, min_value=0, inclusive=False),
                  default=Decimal("1")),
    "max_spread":
        ConfigVar(key="max_spread",
                  prompt="What is the maximum spread? (Enter 1 to indicate 1% or -1 to ignore this setting) >>> ",
                  type_str="decimal",
                  validator=lambda v: validate_decimal(v),
                  default=Decimal("-1")),
    "max_order_age":
        ConfigVar(key="max_order_age",
                  prompt="What is the maximum life time of your orders (in seconds)? >>> ",
                  type_str="float",
                  validator=lambda v: validate_decimal(v, min_value=0, inclusive=False),
                  default=60. * 60.),
    "rsi_period":
        ConfigVar(key="rsi_period",
                  prompt="RSI period. Default is 14",
                  type_str="int",
                  default=14),
    "rsi_overbought":
        ConfigVar(key="rsi_overbought",
                  prompt="RSI Overbought value. Default is 70",
                  type_str="int",
                  default=70),
    "rsi_oversold":
        ConfigVar(key="rsi_oversold",
                  prompt="RSI Oversold value. Default is 30",
                  type_str="int",
                  default=30),
    "rsi_interval":
        ConfigVar(key="rsi_interval",
                  prompt="RSI interval value in seconds. Default is 60 seconds",
                  type_str="int",
                  default=60),
}

