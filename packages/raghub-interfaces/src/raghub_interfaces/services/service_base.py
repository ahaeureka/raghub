from abc import abstractmethod

import grpc
from raghub_core.utils.class_meta import SingletonRegisterMeta


class ServiceBase(metaclass=SingletonRegisterMeta):
    @abstractmethod
    async def initialize(self):
        """
        Initialize the service.
        This method should be overridden in subclasses to perform any necessary initialization.
        """
        raise NotImplementedError("This method should be overridden in subclasses.")

    def get_request_id(self, context: grpc.ServicerContext) -> str:
        """
        Get the request ID from the gRPC context.
        Args:
            context: The gRPC ServicerContext.
        Returns:
            The request ID as a string.
        """
        return context.invocation_metadata().get("request_id", "unknown")
