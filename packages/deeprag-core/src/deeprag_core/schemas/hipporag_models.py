import datetime
from typing import Dict, List, Optional

from pydantic import ConfigDict
from sqlmodel import JSON, Column, Field, SQLModel


class OpenIEInfo(SQLModel, table=True):  # type: ignore[call-arg]
    model_config = ConfigDict(protected_namespaces=())
    __tablename__ = "deeprag_openie_info"
    idx: str = Field(default=None, primary_key=True)
    passage: str = Field(..., description="The passage of text from which OpenIE results are extracted.")
    extracted_entities: List[str] = Field(
        ...,
        description="List of named entities extracted from the text.",
        sa_column=Column(JSON, doc="List of named entities extracted from the text."),
    )
    extracted_triples: List[List[str]] | List[Dict[str, str]] = Field(
        ...,
        description="List of triples extracted from the text.",
        sa_column=Column(JSON, doc="List of triples extracted from the text."),
    )
    created_at: Optional[datetime.datetime] = Field(
        description="Creation time", default=None, sa_column_kwargs={"comment": "creation time"}
    )
    updated_at: Optional[datetime.datetime] = Field(
        description="Update time", default=None, sa_column_kwargs={"comment": "Update time"}
    )
    deleted_at: Optional[datetime.datetime] = Field(
        description="Delete time", default=None, sa_column_kwargs={"comment": "Delete time"}
    )
    is_deleted: Optional[bool] = Field(
        description="Flag to indicate if the record is deleted. Default is False.",
        default=False,
        sa_column_kwargs={"comment": "Flag to indicate if the record is deleted. Default is False"},
    )


class DocumentPassage(SQLModel, table=True):  # type: ignore[call-arg]
    model_config = ConfigDict(protected_namespaces=())
    __tablename__ = "deeprag_document_passage"
    idx: str = Field(default=None, primary_key=True)
    passage: str = Field(..., description="The passage of text from which OpenIE results are extracted.")
    entities: List[str] = Field(
        ...,
        description="List of named entities extracted from the text.",
        sa_column=Column(JSON, doc="List of named entities extracted from the text."),
    )
    triples: List[List[str]] = Field(
        ...,
        description="List of triples extracted from the text.",
        sa_column=Column(JSON, doc="List of triples extracted from the text."),
    )
    created_at: Optional[datetime.datetime] = Field(
        description="Creation time", default=None, sa_column_kwargs={"comment": "creation time"}
    )
    updated_at: Optional[datetime.datetime] = Field(
        description="Update time", default=None, sa_column_kwargs={"comment": "Update time"}
    )
    deleted_at: Optional[datetime.datetime] = Field(
        description="Delete time", default=None, sa_column_kwargs={"comment": "Delete time"}
    )
    is_deleted: Optional[bool] = Field(
        description="Flag to indicate if the record is deleted. Default is False.",
        default=False,
        sa_column_kwargs={"comment": "Flag to indicate if the record is deleted. Default is False"},
    )
