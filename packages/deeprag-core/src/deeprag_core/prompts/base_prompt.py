from abc import ABC, abstractmethod

from langchain_core.prompts import ChatPromptTemplate


class BasePrompt(ABC):
    @abstractmethod
    def get(self, lang="en") -> ChatPromptTemplate:
        raise NotImplementedError("This method should be overridden by subclasses")
