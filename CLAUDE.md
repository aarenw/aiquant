# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个量化交易的软件. 通过 Interactive Brokers Gateway 或 Trader Workstation (TWS) 获取数据和执行交易。

## 开发环境

- **Python 版本**: 3.11+
- **包管理器**: uv
- **虚拟环境**: `.venv/` (使用 uv 管理)

## 常用命令

### 依赖管理
```bash
# 安装依赖
uv sync

# 添加新依赖
uv add <package-name>

# 更新依赖
uv lock --upgrade
```

### 运行项目
```bash
# 运行主程序
uv run python main.py

# 或使用虚拟环境
.venv/bin/python main.py
```



## 架构说明

### 数据采集
数据源来自ikbr，历史行情数据要求全复权，及包含拆股调整和分红复权调整

### 预测模型
用于预测交易产品的趋势，不同的模型放在不同目录，神经网络模型可以参考 `doc/ref/neuronetworksbook.pdf`

### 回测
通过回测评价交易策略

### 模型serving
运行模型，提供restapi以供外部访问。使用flask和python

### 交易策略

### 交易执行




### 本地依赖
- ib_async , 通过ib_async库的api连接 ikbr gateway或者Trader Workstation


### IB 连接配置

- **Client ID**: 123
- **Host**: 127.0.0.1
- **Port**: 7497

#### Trader workstation
port
- Live account: 7496 
- Paper account: 7497
#### Gateway:
- Live account: 4002
- Paper: 4002

在开发前需要确保 Interactive Brokers Gateway 或 Trader Workstation (TWS) 已启动并配置正确的 API 端口。

## 注意事项

- IB API 连接需要本地运行的 IB Gateway 或 TWS,且需要配置允许 API 连接
