from abc import ABC, abstractmethod
from typing import List, Optional

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from raghub_core.prompts.base_prompt import BasePrompt
from raghub_core.schemas.prompt import PromptModel
from raghub_core.schemas.rag_model import RetrieveResultItem


class DefaultQAPrompt(BasePrompt):
    """
    Default prompt for question-answering tasks.
    """

    def __init__(self, prompts: Optional[List[PromptModel]] = None):
        prompts = prompts or [
            PromptModel(
                language="en",
                system_message="""
                        You are a helpful assistant that answers questions based on 
                        the provided context from knowledge base.
                        If the context does not contain relevant information, you will answer "I don't know".
                        """,
                user_message="Context: {context}\nQuestion: {question}\nAnswer:",
            ),
            PromptModel(
                language="zh",
                system_message="""
                        你是一个有用的智能助手，擅长根据知识库提供的上下文回答问题。
                        当上下文中没有相关信息时，你会回答“我不知道”。
                        """,
                user_message="上下文: {context}\n问题: {question}\n答案:",
            ),
        ]
        self._prompts = {prompt.language: prompt for prompt in prompts}

    def get(self, lang="zh") -> ChatPromptTemplate:
        """
        Returns the prompt template for the specified language.
        """
        prompt = self._prompts.get(lang)
        if not prompt:
            raise ValueError(f"No prompt found for language: {lang}")
        system = SystemMessagePromptTemplate.from_template(prompt.system_message)
        user = HumanMessagePromptTemplate.from_template(prompt.user_message)
        # Build the prompt template
        prompt_template = ChatPromptTemplate.from_messages([system, user])
        return prompt_template


class QAPromptBuilder(ABC):
    @abstractmethod
    def build(self, docs: List[RetrieveResultItem], question: str, lang="zh"):
        raise NotImplementedError("This method should be overridden by subclasses.")


class DefaultQAPromptBuilder(QAPromptBuilder):
    def build(self, items: List[RetrieveResultItem], question: str, lang="zh") -> str:
        """
        Builds a question-answering prompt using the provided documents and question.
        """
        context = "\n\n".join([item.document.content for item in items])
        prompt = DefaultQAPrompt().get(lang).invoke(dict(context=context, question=question))
        return prompt.to_string()
