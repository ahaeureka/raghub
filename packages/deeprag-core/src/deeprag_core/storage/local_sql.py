import os
import datetime
from typing import Optional

from deeprag_core.storage.sql import SQLStorage
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine


class LocalSQLStorage(SQLStorage):
    name = "sqlite"

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._engine: Optional[Engine] = None

    def init(self):
        path = self.db_url.split("://")[-1]
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

    def get(self, key: str, model_cls: type[SQLModel]) -> SQLModel | None:
        with Session(self._engine) as session:
            return session.get(model_cls, key)

    def batch_add(self, data: list[SQLModel]):
        with Session(self._engine) as session:
            for item in data:
                session.merge(item)
            session.commit()
