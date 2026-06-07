# signals/sentiment_signal.py
# ============================================================
# SENTIMENT SIGNAL
# Free edge from Reddit + News
# ============================================================

import logging
import aiohttp
from datetime import datetime
from models.data_models import SignalResult

logger = logging.getLogger(__name__)


class SentimentSignal:
    """Reddit + News sentiment signal"""

    def __init__(self):
        self._cache: dict = {}
        self._cache_time: dict = {}

    async def get_reddit_sentiment(
        self,
        symbol: str
    ) -> SignalResult:
        """
        Reddit mention velocity.
        Too many mentions = retail FOMO = contrarian sell
        Very few mentions = accumulation = slight buy
        """
        coin_map = {
            'BTCUSDT':  'bitcoin',
            'ETHUSDT':  'ethereum',
            'SOLUSDT':  'solana',
            'BNBUSDT':  'binance',
            'AVAXUSDT': 'avalanche',
        }
        coin = coin_map.get(symbol, 'bitcoin')

        try:
            # Check cache (5 min)
            cache_key = f"reddit_{symbol}"
            if cache_key in self._cache_time:
                age = (
                    datetime.now() - self._cache_time[cache_key]
                ).seconds
                if age < 300:
                    return self._cache[cache_key]

            url = (
                f"https://www.reddit.com/search.json"
                f"?q={coin}&sort=new&limit=25&t=hour"
            )
            headers = {'User-Agent': 'Mozilla/5.0'}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data   = await resp.json()
                        posts  = data.get(
                            'data', {}
                        ).get('children', [])
                        n = len(posts)

                        # Scoring
                        if n >= 20:
                            score = -0.2  # Too hot = contrarian
                        elif n >= 10:
                            score = -0.1  # Elevated = slight caution
                        elif n <= 2:
                            score = +0.1  # Quiet = accumulation
                        else:
                            score = 0.0   # Normal

                        result = SignalResult(
                            signal_name='reddit_sentiment',
                            value=float(n),
                            score=float(score),
                            weight=0.05,
                            timestamp=datetime.now(),
                            metadata={
                                'posts_per_hour': n,
                                'coin': coin
                            }
                        )

                        self._cache[cache_key]      = result
                        self._cache_time[cache_key] = datetime.now()
                        return result

        except Exception as e:
            logger.debug(f"Reddit signal unavailable: {e}")

        return SignalResult(
            signal_name='reddit_sentiment',
            value=0.0,
            score=0.0,
            weight=0.05,
            timestamp=datetime.now(),
            metadata={'source': 'unavailable'}
        )

    async def get_combined_sentiment(
        self,
        symbol: str
    ) -> float:
        """Combined sentiment score"""
        try:
            reddit = await self.get_reddit_sentiment(symbol)
            return reddit.score
        except Exception:
            return 0.0
