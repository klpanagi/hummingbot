import aiohttp
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from decimal import Decimal
import logging
from hummingbot.core.utils.async_utils import safe_gather, safe_ensure_future

MINER_BASE_URL = "https://api.hummingbot.io/"

class API_ENDPOINTS:
    MARKETS: str = 'bounty/markets'
    LEADERBOARD: str = 'bounty/leaderboard'

s_decimal_0 = Decimal("0")


@dataclass
class CampaignSummary:
    market_id: int = 0
    trading_pair: str = ""
    exchange_name: str = ""
    spread_max: Decimal = s_decimal_0
    payout_asset: str = ""
    liquidity: Decimal = s_decimal_0
    liquidity_usd: Decimal = s_decimal_0
    active_bots: int = 0
    reward_per_wk: Decimal = s_decimal_0
    apy: Decimal = s_decimal_0


class MinerAPIClient:
    """MinerAPIClient.
    """

    _client: Optional["MinerAPIClient"] = None

    @staticmethod
    def create_url(endpoint: str) -> str:
        # TODO: Add params
        return f'{MINER_BASE_URL}{endpoint}'

    @staticmethod
    async def create_request(endpoint: str):
        resp_json: Dict[str, Any] = {}
        async with aiohttp.ClientSession() as client:
            url = MinerAPIClient.create_url(endpoint)
            resp = await client.get(url)
            resp_json = await resp.json()
        return resp_json

    @staticmethod
    def create_request_sync(endpoint: str, on_response: Callable[[Dict[str, Any]], None]):
        task = safe_ensure_future(MinerAPIClient.create_request(endpoint))
        task.add_done_callback(on_response)

    @classmethod
    def client(cls) -> "MinerAPIClient":
        if cls._client is None:
            cls._client = MinerAPIClient()
        return cls._client

    @staticmethod
    async def get_markets() -> Dict[str, Any]:
        return  await MinerAPIClient.create_request(API_ENDPOINTS.MARKETS)

    @staticmethod
    def get_markets_sync(on_response: Callable[[Dict[str, Any]], None]):
        MinerAPIClient.create_request_sync(
            API_ENDPOINTS.MARKETS,
            on_response
        )

    @staticmethod
    async def get_leaderboard() -> Dict[str, Any]:
        return  await MinerAPIClient.create_request(API_ENDPOINTS.LEADERBOARD)

    @staticmethod
    def get_leaderboard_sync(on_response: Callable[[Dict[str, Any]], None]):
        MinerAPIClient.create_request_sync(
            API_ENDPOINTS.LEADERBOARD,
            on_response
        )
