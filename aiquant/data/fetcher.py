"""通过 ib_async 连接 IBKR TWS 下载全复权历史 OHLCV 数据

数据类型：whatToShow="ADJUSTED_LAST"（全复权）
  - 包含拆股（Stock Split）调整
  - 包含分红（Dividend）复权调整
  - 确保历史价格序列连续，不因公司行为事件产生价格跳变

使用方式（async context manager）：
    async with IBDataFetcher() as fetcher:
        await fetcher.fetch_all(symbols)

注意：运行前需确保 IBKR TWS 或 IB Gateway 已启动并开放 API 连接。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from ib_async import IB, Contract, Stock, util

from aiquant.config import IBConfig, DataConfig, RAW_DATA_DIR

logger = logging.getLogger(__name__)


class IBDataFetcher:
    """IBKR 历史数据下载器。

    通过 ib_async 连接 TWS/Gateway，批量下载 S&P 500 成分股的
    全复权日线数据，自动处理 API 限速和断点续传。
    """

    def __init__(
        self,
        ib_config: IBConfig | None = None,
        data_config: DataConfig | None = None,
        output_dir: Path | None = None,
    ):
        """
        Args:
            ib_config:   IB 连接配置（host/port/client_id 等）
            data_config: 数据获取配置（起止年份、复权方式等）
            output_dir:  原始数据保存目录，默认为 data/raw/
        """
        self.ib_config = ib_config or IBConfig()
        self.data_config = data_config or DataConfig()
        self.output_dir = output_dir or RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ib: IB | None = None

    async def connect(self) -> None:
        """建立与 TWS/Gateway 的 API 连接。"""
        self._ib = IB()
        await self._ib.connectAsync(
            host=self.ib_config.host,
            port=self.ib_config.port,
            clientId=self.ib_config.client_id,
            timeout=self.ib_config.timeout,
            readonly=self.ib_config.readonly,
        )
        logger.info(
            "已连接 IB TWS：%s:%d (clientId=%d)",
            self.ib_config.host, self.ib_config.port, self.ib_config.client_id,
        )

    async def disconnect(self) -> None:
        """断开 API 连接。"""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("已断开 IB 连接")

    async def fetch_symbol(
        self, symbol: str, start_year: int | None = None, end_year: int | None = None
    ) -> pd.DataFrame:
        """下载单只股票的全复权日线数据。

        IB API 的 ADJUSTED_LAST 类型只支持 endDateTime=""（当前时间），
        因此通过 durationStr 指定覆盖的总年数。

        Args:
            symbol:     股票代码（如 "AAPL"）
            start_year: 数据起始年份，默认使用 data_config 配置
            end_year:   数据终止年份，默认使用 data_config 配置

        Returns:
            包含 date/open/high/low/close/volume 列的 DataFrame，
            按日期升序排列，已去重。空 DataFrame 表示获取失败。
        """
        start_year = start_year or self.data_config.start_year
        end_year = end_year or self.data_config.end_year

        # 合约定义并验证（SMART 路由自动选择最优交易所）
        contract = Stock(symbol, "SMART", "USD")
        qualified = await self._ib.qualifyContractsAsync(contract)
        if not qualified or qualified[0] is None:
            logger.warning("无法验证合约 %s，跳过", symbol)
            return pd.DataFrame()

        contract = qualified[0]

        # 计算从 start_year 到当前年份的总年数作为 durationStr
        # ADJUSTED_LAST 不支持指定历史 endDateTime，必须用 "" 表示当前时间
        years = datetime.now().year - start_year + 1
        bars = await self._ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",           # 固定为当前时间
            durationStr=f"{years} Y", # 覆盖 start_year 至今的全部数据
            barSizeSetting=self.data_config.bar_size,
            whatToShow=self.data_config.what_to_show,  # "ADJUSTED_LAST"：全复权
            useRTH=self.data_config.use_rth,  # True=仅正常交易时段
            formatDate=1,             # 返回本地时间 datetime 对象
            timeout=self.ib_config.timeout,
        )

        if not bars:
            logger.warning("%s 未返回数据", symbol)
            return pd.DataFrame()

        # 转为 DataFrame，去重并按日期排序
        df = util.df(bars)
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

        # 截取指定时间范围内的数据
        start_date = date(start_year, 1, 1)
        end_date = date(end_year, 12, 31)
        df = df[
            (df["date"] >= pd.Timestamp(start_date)) &
            (df["date"] <= pd.Timestamp(end_date))
        ].reset_index(drop=True)

        return df

    async def fetch_all(
        self,
        symbols: list[str],
        skip_existing: bool = True,
    ) -> dict[str, Path]:
        """批量下载多只股票数据，支持断点续传。

        Args:
            symbols:       股票代码列表
            skip_existing: True=已有 parquet 文件的股票跳过下载

        Returns:
            成功下载的 {symbol: 文件路径} 字典
        """
        saved: dict[str, Path] = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols, 1):
            out_path = self.output_dir / f"{symbol}.parquet"

            # 断点续传：文件已存在则跳过
            if skip_existing and out_path.exists():
                logger.info("[%d/%d] %s 已存在，跳过", i, total, symbol)
                saved[symbol] = out_path
                continue

            logger.info("[%d/%d] 正在下载 %s ...", i, total, symbol)
            try:
                df = await self.fetch_symbol(symbol)
                if df.empty:
                    continue
                df.to_parquet(out_path, index=False)
                saved[symbol] = out_path
                logger.info(
                    "[%d/%d] %s 已保存：%d 条记录（%s ~ %s）",
                    i, total, symbol, len(df),
                    df["date"].iloc[0], df["date"].iloc[-1],
                )
            except Exception as e:
                logger.error("[%d/%d] %s 下载失败：%s", i, total, symbol, e)

        return saved

    async def __aenter__(self):
        """进入 async with 块时自动连接。"""
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        """退出 async with 块时自动断开连接。"""
        await self.disconnect()
