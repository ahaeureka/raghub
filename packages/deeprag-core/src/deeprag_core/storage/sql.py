from abc import abstractmethod

from deeprag_core.utils.class_meta import SingletonRegisterMeta
from sqlmodel import SQLModel


class SQLStorage(metaclass=SingletonRegisterMeta):
    # name = None
    # _registry: Dict[str, Type["SQLStorage"]] = {}  # 注册表
    @abstractmethod
    def init(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def add(self, data: SQLModel):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get(self, key: str) -> SQLModel:
        raise NotImplementedError("Subclasses should implement this method.")

    # @abstractmethod
    # def update(self, key: str, data: SQLModel):
    #     raise NotImplementedError("Subclasses should implement this method.")

    # @abstractmethod
    # def delete(self, key: str):
    #     raise NotImplementedError("Subclasses should implement this method.")

    # @abstractmethod
    # def exec(self, statement: Executable):
    #     raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get_engine(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def batch_add(self, data: list[SQLModel]):
        raise NotImplementedError("Subclasses should implement this `batch_add` method.")
