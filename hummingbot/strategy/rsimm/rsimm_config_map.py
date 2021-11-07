from decimal import Decimal
from typing import Optional

from hummingbot.client.config.config_var import ConfigVar
from hummingbot.client.config.config_validators import (
    validate_exchange,
    validate_decimal,
)

from hummingbot.client.settings import (
    required_exchanges,
)


def exchange_on_validated(value: str) -> None:
    required_exchanges.append(value)


# Returns a market prompt that incorporates the connector value set by the user
def market_prompt() -> str:
    exchange = rsimm_config_map.get("exchange").value
    return f'Enter the token trading pair on {exchange} >>> '


def order_size_prompt() -> str:
    token = rsimm_config_map["token"].value
    return f"What is the size of each order (in {token} amount)? >>> "


def token_validate(value: str) -> Optional[str]:
    value = value.upper()
    market = rsimm_config_map["market"].value
    tokens = market.strip().upper().split("-")
    tokens = [token.strip() for token in tokens]
    if value not in tokens:
        return f"Invalid token. {value} is not one of {','.join(sorted(tokens))}"


# List of parameters defined by the strategy
rsimm_config_map = {
    "strategy":
        ConfigVar(
            key="strategy",
            prompt="",
            default="rsimm",
        ),
    "exchange":
        ConfigVar(key="exchange",
                  prompt="Enter the spot connector to use for liquidity mining >>> ",
                  validator=validate_exchange,
                  on_validated=exchange_on_validated,
                  prompt_on_new=True),
    "market":
        ConfigVar(
            key="market",
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
        ConfigVar(
            key="order_amount",
            prompt=order_size_prompt,
            type_str="decimal",
            prompt_on_new=True
        ),
    "inventory_range_multiplier":
        ConfigVar(key="inventory_range_multiplier",
                  prompt="What is your tolerable range of inventory around the target, expressed in multiples of your total order size? ",
                  type_str="decimal",
                  validator=lambda v: validate_decimal(v, min_value=0, inclusive=False),
                  default=Decimal("1")),
    "rsi_period":
        ConfigVar(key="rsi_period",
                  prompt="RSI period (Default is 14) >>> ",
                  type_str="int",
                  default=14),
    "rsi_overbought":
        ConfigVar(key="rsi_overbought",
                  prompt="RSI Overbought value (Default is 70) >>> ",
                  type_str="int",
                  default=70),
    "rsi_oversold":
        ConfigVar(key="rsi_oversold",
                  prompt="RSI Oversold value (Default is 30) >>> ",
                  type_str="int",
                  default=30),
    "rsi_interval":
        ConfigVar(key="rsi_interval",
                  prompt="RSI interval value in seconds (Default is 60 seconds) >>> ",
                  type_str="int",
                  default=60),
}
