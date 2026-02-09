from src.envs.base import PhysicsControlEnv
from src.envs.oscillator import ForcedOscillator
from src.envs.pendulum import ForcedPendulum
from src.envs.spaceship import ForcedTwoBodySpaceship
from src.envs.three_body import ThreeBodyEnv

ENV_REGISTRY = {
    "oscillator": ForcedOscillator,
    "pendulum": ForcedPendulum,
    "oscillator_visual": ForcedOscillator,
    "pendulum_visual": ForcedPendulum,
    "spaceship": ForcedTwoBodySpaceship,
    "three_body": ThreeBodyEnv,
}

# dm_control wrapper is optional (requires gymnasium + shimmy + dm_control)
try:
    from src.envs.dm_control_wrapper import DMControlPendulumEnv
    ENV_REGISTRY["pendulum_dmcontrol"] = DMControlPendulumEnv
except ImportError:
    pass
