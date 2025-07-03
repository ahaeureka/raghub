"""
RAGHub Server Manager for Testing

This module provides utilities to start and stop the RAGHub server
automatically during test execution.
"""

import asyncio
import logging
import os
import signal
import subprocess
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class RAGHubServerManager:
    """
    Manager for starting and stopping RAGHub server during tests
    """

    def __init__(
        self,
        config_path: str = "test.toml",
        base_url: str = "http://localhost:8000",
        startup_timeout: int = 60,
        health_check_interval: float = 1.0,
    ):
        self.config_path = config_path
        self.base_url = base_url
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval
        self.process: Optional[subprocess.Popen] = None
        self._is_running = False

    async def start_server(self) -> bool:
        """
        Start the RAGHub server

        Returns:
            bool: True if server started successfully, False otherwise
        """
        if self._is_running:
            logger.info("Server is already running")
            return True

        # Check if server is already running
        if await self._is_server_healthy():
            logger.info("Server is already running on another process")
            self._is_running = True
            return True

        logger.info(f"Starting RAGHub server with config: {self.config_path}")

        # Ensure we're in the correct directory
        work_dir = "/app"
        if not os.path.exists(os.path.join(work_dir, self.config_path)):
            logger.error(f"Config file not found: {os.path.join(work_dir, self.config_path)}")
            return False
        print(f"Using config file: {self.config_path}")
        try:
            # Start the server process
            cmd = ["raghub", "start", "server", "-c", self.config_path]
            logger.info(f"Executing command: {' '.join(cmd)} in directory: {work_dir}")

            self.process = subprocess.Popen(
                cmd,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid,  # Create new process group for proper cleanup
            )

            # Wait for server to become healthy
            start_time = time.time()
            while time.time() - start_time < self.startup_timeout:
                if await self._is_server_healthy():
                    logger.info("✅ RAGHub server started successfully")
                    self._is_running = True
                    return True

                # Check if process is still running
                if self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    logger.error(f"Server process exited with code {self.process.returncode}")
                    logger.error(f"STDOUT: {stdout}")
                    logger.error(f"STDERR: {stderr}")
                    return False

                await asyncio.sleep(self.health_check_interval)

            logger.error(f"Server failed to start within {self.startup_timeout} seconds")
            await self.stop_server()
            return False

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            await self.stop_server()
            return False

    async def stop_server(self) -> None:
        """
        Stop the RAGHub server
        """
        if not self._is_running and self.process is None:
            return

        logger.info("Stopping RAGHub server...")

        if self.process:
            try:
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Server didn't stop gracefully, forcing termination")
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait()

                logger.info("✅ RAGHub server stopped")

            except ProcessLookupError:
                logger.warning("Server process was already terminated")
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
            finally:
                self.process = None

        self._is_running = False

    async def _is_server_healthy(self) -> bool:
        """
        Check if the server is healthy by calling health endpoint

        Returns:
            bool: True if server is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    async def wait_for_server(self, timeout: int = 30) -> bool:
        """
        Wait for server to become healthy

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if server becomes healthy, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await self._is_server_healthy():
                return True
            await asyncio.sleep(self.health_check_interval)
        return False

    @property
    def is_running(self) -> bool:
        """Check if the server is currently running"""
        return self._is_running

    async def restart_server(self) -> bool:
        """
        Restart the server

        Returns:
            bool: True if restart successful, False otherwise
        """
        await self.stop_server()
        await asyncio.sleep(2)  # Give some time for cleanup
        return await self.start_server()


# Global server manager instance
_server_manager: Optional[RAGHubServerManager] = None


async def get_server_manager(rag_mode: Optional[str] = None) -> RAGHubServerManager:
    """
    Get or create the global server manager instance

    Args:
        rag_mode: Optional RAG mode to use for server configuration

    Returns:
        RAGHubServerManager: The server manager instance
    """
    global _server_manager
    if _server_manager is None:
        from .config import TestConfig

        config_path = TestConfig.get_server_config_path(rag_mode)

        _server_manager = RAGHubServerManager(
            config_path=config_path,
            base_url=TestConfig.BASE_URL,
            startup_timeout=TestConfig.SERVER_STARTUP_TIMEOUT,
        )
    return _server_manager


async def ensure_server_running(rag_mode: Optional[str] = None) -> bool:
    """
    Ensure the RAGHub server is running

    Args:
        rag_mode: Optional RAG mode to use for server configuration

    Returns:
        bool: True if server is running, False otherwise
    """
    manager = await get_server_manager(rag_mode)
    if not manager.is_running:
        return await manager.start_server()
    return True


async def cleanup_server() -> None:
    """
    Clean up the server manager and stop the server
    """
    global _server_manager
    if _server_manager is not None:
        await _server_manager.stop_server()
        _server_manager = None
