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
        metadata = context.invocation_metadata()
        request_id = next((value for key, value in metadata if key.lower() == "x-request-id"), None)
        return request_id

    def get_auth(self, context: grpc.ServicerContext) -> str:
        """
        Get the authentication token from the gRPC context.
        Args:
            context: The gRPC ServicerContext.
        Returns:
            The authentication token as a string.
        """
        auth_metadata = context.invocation_metadata()
        return next((value for key, value in auth_metadata if key.lower() == "authorization"), None)
