import datetime
import os
from typing import Any, List, Optional, Type

from deeprag_core.storage.structed_data import StructedDataStorage
from loguru import logger
from sqlalchemy import Engine, Executable
from sqlmodel import Session, SQLModel, create_engine, select, update


class LocalSQLStorage(StructedDataStorage):
    name = "sqlite"

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._engine: Optional[Engine] = None

    def init(self):
        path = self.db_url.split("://")[-1]
        logger.debug(f"Initializing SQLite database at {self.db_url}")
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)
        self._engine = create_engine(self.db_url)
        SQLModel.metadata.create_all(self._engine)

    def get_engine(self) -> Engine:
        if self._engine is None:
            raise ValueError("Engine not initialized. Call init() first.")
        return self._engine

    def close(self):
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def add(self, data: SQLModel):
        with Session(self._engine) as session:
            if hasattr(data, "created_at"):
                data.created_at = datetime.datetime.now(datetime.timezone.utc)
            session.merge(data)
            session.commit()

    def get(self, keys: List[str], model_cls: type[SQLModel]) -> List[SQLModel]:
        with Session(self._engine) as session:
            primary_key_names = self.get_primary_key_names(model_cls)
            if not primary_key_names:
                raise ValueError(f"Model {model_cls.__name__} does not have a primary key.")
            if len(primary_key_names) > 1:
                raise ValueError(
                    f"Model {model_cls.__name__} has a composite primary key. Please provide a list of keys."
                )
            if not keys:
                raise ValueError("Keys list is empty.")
            sql = select(model_cls).where(getattr(model_cls, primary_key_names[0]).in_(keys))
            if hasattr(model_cls, "is_deleted"):
                sql = sql.where(model_cls.is_deleted == False)  # noqa: E712
            logger.debug(f"SQL: {sql.compile(compile_kwargs={'literal_binds': True})}")
            ret = session.exec(sql).fetchall()
            return ret

            # return session.exec(model_cls).filter(getattr(model_cls, model_cls.__primary_key__).__eq__(keys)).all()

    def delete(self, keys: list[str], model_cls: Type[SQLModel]):
        primary_key_names = self.get_primary_key_names(model_cls)
        if not primary_key_names:
            raise ValueError(f"Model {model_cls.__name__} does not have a primary key.")
        if len(primary_key_names) > 1:
            raise ValueError(f"Model {model_cls.__name__} has a composite primary key. Please provide a list of keys.")
        if not keys:
            raise ValueError("Keys list is empty.")

        self.update(
            model_cls,
            {"deleted_at": datetime.datetime.now(datetime.timezone.utc), "is_deleted": True},
            getattr(model_cls, primary_key_names[0]).in_(keys),
        )

    def update(self, model_cls: Type[SQLModel], updated: dict, *where):
        with Session(self._engine) as session:
            statement = update(model_cls).values(**updated).where(*where)
            session.exec(statement)
            session.commit()

        return True

    def batch_add(self, data: list[SQLModel]):
        with Session(self._engine) as session:
            for item in data:
                if hasattr(item, "created_at"):
                    item.created_at = datetime.datetime.now(datetime.timezone.utc)
                session.merge(item)
            session.commit()

    def exec(self, statement: Executable) -> Any:
        with Session(self._engine) as session:
            result = session.exec(statement)
            return result
