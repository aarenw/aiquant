"""模型训练脚本"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aiquant.config import DataConfig, TrainConfig, RAW_DATA_DIR
from aiquant.data.features import engineer_features, get_feature_columns
from aiquant.data.preprocessing import prepare_single_stock, time_based_split
from aiquant.data.dataset import create_dataloaders
from aiquant.models.transformer_trend import TrendTransformer, TransformerConfig
from aiquant.training.trainer import Trainer, compute_class_weights
from aiquant.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_process_data(
    data_dir: Path, symbols: list[str] | None = None, data_config: DataConfig | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data_config = data_config or DataConfig()

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if symbols:
        parquet_files = [f for f in parquet_files if f.stem in symbols]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    logger.info("Processing %d symbols...", len(parquet_files))

    all_dfs = []
    feature_cols = None

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        df = engineer_features(df)

        if feature_cols is None:
            feature_cols = get_feature_columns(df)

        df = prepare_single_stock(df, feature_cols, data_config)
        if len(df) > data_config.seq_len:
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid data after processing")

    logger.info("Processed %d symbols, total rows: %d", len(all_dfs), sum(len(d) for d in all_dfs))

    combined = pd.concat(all_dfs, ignore_index=True)
    split = time_based_split(combined, feature_cols, config=data_config)

    logger.info(
        "Split sizes - train: %d, val: %d, test: %d",
        len(split.y_train), len(split.y_val), len(split.y_test),
    )

    return (
        split.X_train, split.y_train,
        split.X_val, split.y_val,
        split.X_test, split.y_test,
        feature_cols,
    )


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    data_config = DataConfig(seq_len=args.seq_len)
    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        seed=args.seed,
    )

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = (
        load_and_process_data(Path(args.data_dir), data_config=data_config)
    )

    n_features = X_train.shape[2]
    logger.info("Features: %d, Seq length: %d", n_features, args.seq_len)

    class_weights = compute_class_weights(y_train)
    logger.info("Class weights: %s", class_weights.tolist())

    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=train_config.batch_size,
    )

    model_config = TransformerConfig(
        n_features=n_features,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    model = TrendTransformer(model_config)
    logger.info("Model parameters: %s", f"{model.count_parameters():,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        class_weights=class_weights,
    )

    history = trainer.train()

    logger.info(
        "Training complete. Best epoch: %d, Best F1: %.4f",
        history.best_epoch, history.best_metric,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrendTransformer model")
    parser.add_argument("--data-dir", default=str(RAW_DATA_DIR))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
