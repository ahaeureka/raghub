from pydantic import BaseModel


class HistoryContextItem(BaseModel):
    role: str
    content: str


class HistoryContext(BaseModel):
    items: list[HistoryContextItem] = []
    """
    Represents the history context of a conversation.
    
    Attributes:
        items (list[HistoryContextItem]): List of items in the history context.
    """

    def add_item(self, role: str, content: str):
        """Adds an item to the history context."""
        self.items.append(HistoryContextItem(role=role, content=content))
