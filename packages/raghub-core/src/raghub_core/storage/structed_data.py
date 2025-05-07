from abc import abstractmethod
from typing import Any, List, Type

from raghub_core.utils.class_meta import SingletonRegisterMeta
from sqlalchemy import Executable
from sqlmodel import SQLModel


class StructedDataStorage(metaclass=SingletonRegisterMeta):
    @abstractmethod
    def init(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def add(self, data: SQLModel):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get(self, keys: List[str], model_cls: type[SQLModel]) -> List[SQLModel]:
        raise NotImplementedError("Subclasses should implement this method.")

    # @abstractmethod
    # def update(self, key: str, data: SQLModel):
    #     raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def delete(self, keys: List[str], model_cls: type[SQLModel]):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def exec(self, statement: Executable) -> Any:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get_engine(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def batch_add(self, data: list[SQLModel]):
        raise NotImplementedError("Subclasses should implement this `batch_add` method.")

    @classmethod
    def get_primary_key_names(cls, model_class: Type[SQLModel]) -> List[str]:
        """
        Get the primary key names of a SQLModel class.
        Args:
            model_class (Type[SQLModel]): The SQLModel class to inspect.
        Returns:
            List[str]: A list of primary key names.
        """
        primary_key_names = [col.name for col in model_class.__table__.primary_key.columns.values()]
        return primary_key_names
