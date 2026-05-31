"""数据下载脚本 - 通过 IB TWS 批量获取 S&P 500 全复权日线数据"""

import argparse
import asyncio
import logging
import sys

from aiquant.config import IBConfig, DataConfig
from aiquant.data.symbols import get_symbols
from aiquant.data.fetcher import IBDataFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main(args: argparse.Namespace) -> None:
    symbols = get_symbols()
    if args.symbols:
        symbols = args.symbols.split(",")

    if args.limit:
        symbols = symbols[: args.limit]

    logger.info("Fetching %d symbols", len(symbols))

    ib_config = IBConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
    )
    data_config = DataConfig(
        start_year=args.start_year,
        end_year=args.end_year,
    )

    async with IBDataFetcher(ib_config=ib_config, data_config=data_config) as fetcher:
        saved = await fetcher.fetch_all(symbols, skip_existing=not args.force)

    logger.info("Done. Saved %d / %d symbols.", len(saved), len(symbols))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical data from IB TWS")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=123)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--symbols", type=str, help="Comma-separated symbol list")
    parser.add_argument("--limit", type=int, help="Limit number of symbols")
    parser.add_argument("--force", action="store_true", help="Re-download existing data")

    args = parser.parse_args()
    asyncio.run(main(args))
