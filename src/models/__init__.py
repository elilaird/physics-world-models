from src.models.discrete import JumpModel, LSTMModel
from src.models.ode import FirstOrderODENet, NewtonianDynamicsModel, VelocityDynamicsModel
from src.models.hamiltonian import PortHamiltonianModel
from src.models.wrappers import TrajectoryMatchingModel, TrajectoryShootingModel

MODEL_REGISTRY = {
    "jump": JumpModel,
    "lstm": LSTMModel,
    "first_order_ode": FirstOrderODENet,
    "newtonian": NewtonianDynamicsModel,
    "velocity": VelocityDynamicsModel,
    "port_hamiltonian": PortHamiltonianModel,
}

WRAPPER_REGISTRY = {
    "trajectory_matching": TrajectoryMatchingModel,
    "trajectory_shooting": TrajectoryShootingModel,
}
