from src.envs.base import PhysicsControlEnv
from src.envs.oscillator import ForcedOscillator
from src.envs.pendulum import ForcedPendulum

ENV_REGISTRY = {
    "oscillator": ForcedOscillator,
    "pendulum": ForcedPendulum,
    "oscillator_visual": ForcedOscillator,
    "pendulum_visual": ForcedPendulum,
}
