import pytest


def test_training_requires_a_fine_tuning_epoch():
    from src.train import train
    from src.utils import Config

    cfg = Config(num_epochs=5, freeze_epochs=5)

    with pytest.raises(ValueError, match="backbone is fine-tuned"):
        train(cfg)
