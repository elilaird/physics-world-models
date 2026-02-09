from src.envs.base import PhysicsControlEnv
from src.envs.oscillator import ForcedOscillator
from src.envs.pendulum import ForcedPendulum
from src.envs.spaceship import ForcedTwoBodySpaceship
from src.envs.three_body import ThreeBodyEnv

ENV_REGISTRY = {
    "oscillator": ForcedOscillator,
    "pendulum": ForcedPendulum,
    "spaceship": ForcedTwoBodySpaceship,
    "three_body": ThreeBodyEnv,
}
