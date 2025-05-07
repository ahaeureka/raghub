from abc import ABC, abstractmethod
from typing import Any

from raghub_app.app_schemas.app import APPRunnerParams


class BaseAPP(ABC):
    name: str
    description: str

    @abstractmethod
    def initialization(self):
        raise NotImplementedError("Subclasses must implement the initialization method")

    @abstractmethod
    def run(self, input: APPRunnerParams) -> Any:
        raise NotImplementedError("Subclasses must implement the run method")

    @abstractmethod
    async def arun(self, input: APPRunnerParams) -> Any:
        raise NotImplementedError("Subclasses must implement the async_run method")
