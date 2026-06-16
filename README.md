# aiquant — 基于注意力机制的股票趋势预测

通过 Interactive Brokers TWS 获取 S&P 500 全复权日线数据，使用从零实现的 Transformer Encoder（Multi-Head Self-Attention）预测未来 下一 个交易日的价格趋势（由涨转跌， 维持趋势， 由跌转涨）。

---

## 项目结构

```text
aiquant/
├── aiquant/                        # 核心 Python 包
│   ├── config.py                   # 全局配置（路径/设备/IB连接/训练超参数）
│   ├── data/
│   │   ├── symbols.py              # S&P 500 成分股列表管理
│   │   ├── fetcher.py              # ib_async 连接 TWS 下载全复权 OHLCV → parquet
│   │   ├── indicators.py           # ta-lib 技术指标计算（11类共18列）
│   │   ├── features.py             # 特征工程管线（约33维特征）
│   │   ├── preprocessing.py        # 趋势标签、滚动Z-score标准化、滑动窗口、时间分割
│   │   └── dataset.py              # PyTorch Dataset / DataLoader 封装
│   ├── models/
│   │   └── transformer_trend/
│   │       ├── config.py           # TransformerConfig 超参数 dataclass
│   │       ├── attention.py        # Scaled Dot-Product + Multi-Head Attention（从零实现）
│   │       ├── positional.py       # 正弦位置编码
│   │       ├── encoder.py          # Encoder Layer（MHA→Add&Norm→FFN→Add&Norm）+ Stack
│   │       └── model.py            # TrendTransformer 完整模型
│   ├── training/
│   │   ├── metrics.py              # Accuracy / F1 / Precision / Recall / 混淆矩阵
│   │   ├── checkpoint.py           # 模型保存 / 加载 / 自动清理旧文件
│   │   └── trainer.py              # 训练循环（AdamW + CosineAnnealing + 早停）
│   ├── evaluation/
│   │   ├── evaluator.py            # 测试集推理与指标计算
│   │   └── visualization.py        # 混淆矩阵 / 训练曲线 / 注意力热力图
│   └── utils/
│       └── reproducibility.py      # 随机种子固定
├── scripts/
│   ├── fetch_data.py               # 数据下载脚本（需 TWS 在线）
│   ├── train_model.py              # 模型训练脚本
│   └── evaluate_model.py           # 模型评估脚本
├── tests/
│   ├── test_data/                  # 数据管线单元测试
│   ├── test_models/                # 模型单元测试
│   └── test_training/              # 训练管线单元测试
├── data/                           # 数据目录（.gitignore 不提交）
│   ├── raw/                        # 原始 parquet 文件（每股一个）
│   ├── processed/                  # 处理后数据
│   └── cache/                      # 股票列表缓存
├── checkpoints/                    # 模型权重（.gitignore 不提交）
├── pyproject.toml
└── CLAUDE.md
```

---

## 模型架构

```text
输入 (batch, 60, 33)   — 60 个交易日 × 33 维特征
      ↓  Linear 输入投影
(batch, 60, 64)         — 映射到模型维度 d_model=64
      ↓  正弦位置编码
(batch, 60, 64)
      ↓  Transformer Encoder × 3 层
         每层：Multi-Head Attention (h=4) → Add&Norm → FFN (d_ff=256) → Add&Norm
(batch, 60, 64)
      ↓  时间维度均值池化
(batch, 64)
      ↓  MLP 分类头（Linear → GELU → Dropout → Linear）
(batch, 3)              — 输出 logits：0=由涨转跌 / 1=维持趋势 / 2=由跌转涨
```

**参数量**：默认配置约 120K 参数，可在单 GPU/MPS 上快速训练。

---

## 数据说明

| 项目 | 说明 |
| ---- | ---- |
| 数据源 | Interactive Brokers TWS（`ADJUSTED_LAST` 全复权） |
| 复权方式 | 全复权：含拆股 + 分红调整，历史价格序列连续 |
| 股票范围 | S&P 500 成分股（约 500 只，覆盖 NYSE + NASDAQ） |
| 时间跨度 | 2010 年至今 |
| K 线周期 | 日线，仅正常交易时段（useRTH=True） |
| 特征数量 | 约 33 维（价格 + 技术指标 + 时间特征） |
| 预测目标 | 下一交易日趋势：ZigZag 过滤噪声后，target2 符号翻转（由涨转跌 / 维持 / 由跌转涨） |

---

## 环境准备

**前置要求**：Python 3.11+，[uv](https://github.com/astral-sh/uv) 包管理器。

```bash
# 安装依赖
uv sync --all-extras

# 安装 ta-lib 系统库（macOS）
brew install ta-lib
```

---

## 快速开始

### 第一步：下载历史数据

确保 **IBKR Trader Workstation（TWS）已启动**并在 `全局配置 → API → 设置` 中开启 Socket 客户端，然后执行：

```bash
# 下载全部 S&P 500 成分股（约 500 只），自动断点续传
uv run python scripts/fetch_data.py

# 仅下载指定股票（用于快速测试）
uv run python scripts/fetch_data.py --symbols AAPL,MSFT,GOOGL,AMZN,NVDA

# 指定连接参数（默认：127.0.0.1:7497）
uv run python scripts/fetch_data.py --host 127.0.0.1 --port 7497 --client-id 123

# 强制重新下载（覆盖已有文件）
uv run python scripts/fetch_data.py --force
```

下载完成后原始数据保存在 `data/raw/{SYMBOL}.parquet`。

### 第二步：训练模型

```bash
# 使用默认超参数训练
uv run python scripts/train_model.py

# 自定义主要超参数
uv run python scripts/train_model.py \
    --batch-size 256 \
    --max-epochs 100 \
    --lr 1e-3 \
    --seq-len 60 \
    --d-model 64 \
    --n-heads 4 \
    --n-layers 3
```

训练过程中：

- 每当验证集 Macro F1 创新高，自动保存 checkpoint 到 `checkpoints/`
- 10 轮无提升触发早停
- 日志实时输出每轮的 loss / accuracy / macro_f1 / 学习率

### 第三步：评估模型

```bash
# 在测试集上评估（自动使用最新 checkpoint）
uv run python scripts/evaluate_model.py

# 指定 checkpoint 文件
uv run python scripts/evaluate_model.py \
    --checkpoint checkpoints/model_epoch042_f10.3812.pt \
    --output-dir evaluation_results/
```

输出内容：

- 分类报告（各类别 Precision / Recall / F1）
- 总体准确率和 Macro F1
- 混淆矩阵图片（保存到 `evaluation_results/confusion_matrix.png`）

---

## 配置说明

所有默认值均在 [aiquant/config.py](aiquant/config.py) 中定义：

```python
# IB 连接配置
IBConfig(
    host="127.0.0.1",
    port=7497,      # 7497=TWS模拟盘, 7496=TWS实盘, 4002=IB Gateway
    client_id=123,
    readonly=True,
)

# 数据配置
DataConfig(
    start_year=2010,
    what_to_show="ADJUSTED_LAST",  # 全复权
    seq_len=60,                    # 60日回看窗口
    zigzag_depth=5,                # ZigZag Depth（极值搜索窗口）
    zigzag_backstep=3,             # ZigZag Backstep（极值最小间隔）
    trend_threshold=0.01,          # Deviation 固定回退比例
    train_end="2020-12-31",
    val_end="2022-12-31",
)

# 模型配置
TransformerConfig(
    d_model=64, n_heads=4, n_layers=3, d_ff=256,
    dropout=0.1, seq_len=60, n_classes=3,
)

# 训练配置
TrainConfig(
    batch_size=256, max_epochs=100, lr=1e-3,
    patience=10, max_grad_norm=1.0,
)
```

---

## 运行测试

```bash
# 运行全部单元测试（不需要 TWS 连接）
uv run pytest tests/ -v

# 跳过需要网络的慢速测试
uv run pytest tests/ -v -m "not slow"

# 查看测试覆盖率
uv run pytest tests/ --cov=aiquant --cov-report=term-missing
```

---

## 开发工具

```bash
# 添加新依赖
uv add <package-name>

# 安装开发依赖（pytest 等）
uv sync --all-extras
```

---

## IB TWS 配置

1. 打开 Trader Workstation，登录模拟或实盘账户
2. 进入 `全局配置 → API → 设置`
3. 勾选 **启用 ActiveX 和套接字客户端**
4. 确认端口号与 `IBConfig.port` 一致（模拟盘默认 `7497`）
5. 将 `127.0.0.1` 加入受信任的 IP 地址列表

---

## 技术依赖

| 库 | 用途 |
| -- | ---- |
| [ib_async](https://github.com/ib-api-reloaded/ib_async) | IBKR API 异步客户端 |
| [PyTorch](https://pytorch.org) | 深度学习框架（支持 MPS/CUDA/CPU） |
| [ta-lib](https://github.com/mrjbq7/ta-lib) | 技术指标计算 |
| [pandas](https://pandas.pydata.org) | 数据处理 |
| [scikit-learn](https://scikit-learn.org) | 评估指标 |
| [matplotlib](https://matplotlib.org) + [seaborn](https://seaborn.pydata.org) | 可视化 |
| [pyarrow](https://arrow.apache.org/docs/python) | Parquet 文件读写 |
