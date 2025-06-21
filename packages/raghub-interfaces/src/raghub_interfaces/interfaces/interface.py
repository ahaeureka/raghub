from abc import ABC

from raghub_interfaces.config import interface_config
from raghub_interfaces.schemes.params import RunnerParams


class BaseInterface(ABC):
    name: str
    description: str

    def __init__(self, config: interface_config.InerfaceConfig):
        """
        Initializes the BaseInterface with a configuration.

        :param config: The configuration for the interface.
        """
        self._config = config

    async def __call__(self, params: RunnerParams = None):
        """
        The main entry point for the interface.
        This method should be overridden in subclasses to implement the interface logic.
        """
        raise NotImplementedError("This method should be overridden in subclasses.")
