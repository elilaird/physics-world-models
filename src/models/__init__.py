from .tcn_model import TinyTCN
from .lightweight_io import LightweightIO
from .lstm_model import CompactLSTM
from .mobilenet_1d import MobileNet1D
from .quantized_model import QuantizedModelWrapper


def build_model(config):
    """Build a model from configuration.

    Args:
        config: ModelConfig with name, input_channels, hidden_channels, etc.

    Returns:
        nn.Module instance.
    """
    builders = {
        "tiny_tcn": lambda c: TinyTCN(
            input_channels=c.input_channels,
            hidden_channels=c.hidden_channels,
            num_blocks=c.num_blocks,
            output_dim=c.output_dim,
            dropout=c.dropout,
        ),
        "compact_lstm": lambda c: CompactLSTM(
            input_channels=c.input_channels,
            hidden_size=c.hidden_channels,
            num_layers=c.num_blocks,
            output_dim=c.output_dim,
            dropout=c.dropout,
        ),
        "mobilenet_1d": lambda c: MobileNet1D(
            input_channels=c.input_channels,
            initial_channels=c.hidden_channels,
            num_blocks=c.num_blocks,
            output_dim=c.output_dim,
            dropout=c.dropout,
        ),
        "lightweight_io": lambda c: LightweightIO(
            input_channels=c.input_channels,
            hidden_channels=c.hidden_channels,
            num_blocks=c.num_blocks,
            output_dim=c.output_dim,
            dropout=c.dropout,
        ),
    }

    if config.name not in builders:
        raise ValueError(f"Unknown model: {config.name}. Choose from {list(builders.keys())}")

    return builders[config.name](config)
