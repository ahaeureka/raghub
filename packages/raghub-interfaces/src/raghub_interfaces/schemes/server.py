from typing import Optional

from pydantic import Field
from raghub_interfaces.schemes.params import RunnerParams


class ServerParams(RunnerParams):
    host: Optional[str] = Field(default=None, description="The host address for the server.")
    port: Optional[int] = Field(default=0, description="The port number for the server.")
