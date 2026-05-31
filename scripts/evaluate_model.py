"""模型评估脚本"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from aiquant.config import DataConfig, RAW_DATA_DIR, CHECKPOINT_DIR
from aiquant.data.dataset import TrendDataset
from aiquant.evaluation.evaluator import evaluate_from_checkpoint
from aiquant.evaluation.visualization import plot_confusion_matrix, plot_attention_weights
from aiquant.models.transformer_trend import TrendTransformer, TransformerConfig
from aiquant.utils.reproducibility import set_seed
from scripts.train_model import load_and_process_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def find_best_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = sorted(ckpt_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return ckpts[-1]


def main(args: argparse.Namespace) -> None:
    set_seed(42)
    device = torch.device(args.device)

    data_config = DataConfig(seq_len=args.seq_len)
    _, _, _, _, X_test, y_test, feature_cols = load_and_process_data(
        Path(args.data_dir), data_config=data_config
    )

    n_features = X_test.shape[2]
    model_config = TransformerConfig(
        n_features=n_features,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    model = TrendTransformer(model_config)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(CHECKPOINT_DIR)
    logger.info("Using checkpoint: %s", ckpt_path)

    test_ds = TrendDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    metrics, y_true, y_pred = evaluate_from_checkpoint(model, ckpt_path, test_loader, device)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(metrics.report)
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Macro F1: {metrics.macro_f1:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(metrics, save_path=output_dir / "confusion_matrix.png")
    logger.info("Saved confusion matrix to %s", output_dir / "confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--data-dir", default=str(RAW_DATA_DIR))
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)

    args = parser.parse_args()
    main(args)
