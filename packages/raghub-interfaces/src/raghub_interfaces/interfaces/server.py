import asyncio
import os

from grpc_fastapi_gateway.gateway import Gateway
from hypercorn.asyncio import serve
from hypercorn.config import Config
from loguru import logger

from raghub_interfaces.api.web import WebAPI
from raghub_interfaces.config import interface_config
from raghub_interfaces.interfaces.interface import BaseInterface
from raghub_interfaces.registry.register import Registry
from raghub_interfaces.schemes.server import ServerParams


@Registry.register()
class Server(BaseInterface):
    """
    A class representing a server in the Raghub system.
    """

    name: str = "server"
    description: str = "RAGHub Server Interface"

    def __init__(self, config: interface_config.InerfaceConfig):
        """
        Initializes the Server with a name and address.

        :param name: The name of the server.
        :param address: The address of the server.
        """
        self._web = WebAPI(config)
        super().__init__(config)
        self._config = config
        self._gw: Gateway = None

    def __repr__(self):
        return f"Server(name={self.name}, description={self.description})"

    async def initialize(self):
        """
        Initializes the server.
        This method can be used to perform any startup tasks, such as setting up routes or middleware.
        """
        logger.info("🚀 Initializing RAGHub Server...")

        try:
            await self._web.initialize()
            logger.info("✅ WebAPI initialized successfully")

            from raghub_protos.models import rag_model

            from raghub_interfaces.services.rag import RAGServiceImpl

            services = {"rag": [RAGServiceImpl(self._config)]}
            tasks = []
            for _, service_list in services.items():
                for service in service_list:
                    tasks.append(service.initialize())

            logger.info(f"🔧 Initializing {len(tasks)} services...")
            await asyncio.gather(*tasks)
            logger.info("✅ All services initialized successfully")

            protos_dir = os.path.dirname(os.path.dirname(rag_model.__file__))
            self._gw = Gateway(
                fastapi_app=self._web.app,
                service_groups=services,
                models_dir=os.path.join(protos_dir, "models"),
                pb_dir=os.path.join(protos_dir, "pb"),
                logger=logger,
            )

            logger.info("🔧 Loading Gateway services...")
            self._gw.load_services()
            logger.info("✅ Gateway services loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize server: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    async def __call__(self, input: ServerParams = None):
        """
        Starts the server.
        This method should be overridden in subclasses to implement the server startup logic.
        """
        try:
            await self.initialize()

            config = Config()
            if input is None:
                input = ServerParams()

            host = input.host or self._config.interfaces.server.address
            port = input.port or self._config.interfaces.server.port
            config.bind = [f"{host}:{port}"]

            # Enhanced server configuration for better debugging
            config.workers = 4
            config.loglevel = "DEBUG"
            config.access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
            config.errorlog = "-"  # Log to stdout
            config.accesslog = "-"  # Log to stdout
            config.keep_alive_timeout = 65
            config.graceful_timeout = 30
            config.max_requests = 1000
            config.max_requests_jitter = 50

            # Enable detailed error logging

            if not self._gw:
                raise RuntimeError("Gateway is not initialized.")

            logger.info(f"🚀 Starting server at {host}:{port} with {config.workers} workers.")
            logger.info(f"📚 API documentation available at http://{host}:{port}/docs")
            logger.info(f"🔍 Health check available at http://{host}:{port}/health")
            logger.info(f"🔧 Diagnostic info available at http://{host}:{port}/diagnostic")

            await serve(self._gw, config)  # type: ignore[arg-type]

        except Exception as e:
            logger.error(f"❌ Failed to start server: {type(e).__name__}: {str(e)}", exc_info=True)
            raise
