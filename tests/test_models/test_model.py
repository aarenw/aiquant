import torch
import pytest

from aiquant.models.transformer_trend import TrendTransformer, TransformerConfig


class TestTrendTransformer:
    @pytest.fixture
    def model(self):
        config = TransformerConfig(n_features=33, d_model=32, n_heads=4, n_layers=2)
        return TrendTransformer(config)

    def test_forward_shape(self, model):
        x = torch.randn(4, 60, 33)
        out = model(x)
        assert out.shape == (4, 3)

    def test_different_batch_sizes(self, model):
        for batch in [1, 2, 8, 16]:
            x = torch.randn(batch, 60, 33)
            out = model(x)
            assert out.shape == (batch, 3)

    def test_parameter_count_reasonable(self, model):
        n_params = model.count_parameters()
        assert 1000 < n_params < 2_000_000

    def test_attention_weights_available(self, model):
        x = torch.randn(2, 60, 33)
        model(x)
        weights = model.get_attention_weights()
        assert len(weights) == 2  # n_layers

    def test_overfit_single_batch(self):
        config = TransformerConfig(n_features=10, d_model=16, n_heads=2, n_layers=1)
        model = TrendTransformer(config)

        x = torch.randn(8, 60, 10)
        y = torch.randint(0, 3, (8,))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.CrossEntropyLoss()

        initial_loss = None
        for step in range(50):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()

        assert loss.item() < initial_loss * 0.5
