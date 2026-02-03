"""Environment factory class. Given a valid environment name and its constructor args, returns an instantiation of it
"""
import os
import sys
from typing import Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import Environment
from pendulum import Pendulum
from spring import Spring
from gravity import NObjectGravity
from chaotic_pendulum import ChaoticPendulum


class EnvFactory():
    """Return a new Environment"""

    # Map the name of the name of the Environment concrete class by retrieving all its subclasses
    _name_to_env = {cl.__name__: cl for cl in Environment.__subclasses__()}

    @staticmethod
    def get_environment(name: str, device: Union[str, None] = None, **kwargs):
        """Return an environment object based on the environment identifier.

        Args:
            name (str): name of the class of the concrete Environment.
            device (Union[str, None], optional): Device to run computations on ('cpu' or 'cuda'). 
                Defaults to None (will use 'cpu').
            **kwargs: args supplied to the constructor of the object of class name. 
        
        Raises:
            NameError: if the given environment type is not supported.
        
        Returns:
            Environment: concrete instantiation of the Environment.
        """
        if device is None:
            device = 'cpu'
        
        if 'device' not in kwargs:
            kwargs['device'] = device
        
        try:
            return EnvFactory._name_to_env[name](**kwargs)
        except KeyError:
            available = ", ".join(EnvFactory._name_to_env.keys())
            msg = f"{name} is not a supported type by Environment. Available types are: {available}"
            raise NameError(msg)


if __name__ == "__main__":
    # EnvFactory test
    import torch
    import numpy as np
    from environment import visualize_rollout
    
    env = EnvFactory.get_environment("Pendulum", mass=0.5, length=1, g=10, device='cpu')
    print(type(env))

    rolls = env.sample_random_rollouts(number_of_frames=100,
                                       delta_time=0.1,
                                       number_of_rollouts=16,
                                       img_size=32,
                                       color=False,
                                       noise_level=0.,
                                       seed=23)
    if isinstance(rolls, torch.Tensor):
        rolls = rolls.cpu().numpy()
    idx = np.random.randint(rolls.shape[0])
    visualize_rollout(rolls[idx])