from typing import List, Optional

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from raghub_core.prompts.base_prompt import BasePrompt
from raghub_core.schemas.keywords_model import KeywordsPromptModel


class KeywordsPrompts(BasePrompt):
    """Prompt for extracting keywords from text."""

    name = "KeywordsPrompts"
    description = "Prompt for extracting keywords from text"

    def __init__(self, prompts: Optional[List[KeywordsPromptModel]] = None):
        prompts = prompts or [
            KeywordsPromptModel(
                language="en",
                system_message="A question is provided below. Given the question, extract up to "
                "keywords from the text. Focus on extracting the keywords that we can use "
                "to best lookup answers to the question.\n"
                "Generate as many as possible synonyms or aliases of the keywords "
                "considering possible cases of capitalization, pluralization, "
                "common expressions, etc.\n"
                "For person keywords, include their roles or professions.\n"
                "Avoid stopwords.\n"
                "Provide the keywords and synonyms in comma-separated format."
                "Formatted keywords and synonyms text should be separated by a semicolon.\n"
                "---------------------\n"
                "Example:\n"
                "Text: Alice is Bob's mother.\n"
                "Keywords:\nAlice,mother,Bob;mummy\n"
                "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
                "Keywords:\nPhilz,coffee shop,Berkeley,1982;coffee bar,coffee house\n"
                "---------------------\n",
                user_message="Please extract keywords from the following text: {text}",
            ),
            KeywordsPromptModel(
                language="zh",
                system_message="以下是一个问题。根据问题，从文本中提取尽可能多的关键词。重点提取可用于查找问题答案的关键词。\n"
                "生成尽可能多的关键词的同义词或别名，考虑大小写、复数形式、常用表达等可能情况。\n"
                "避免使用停用词（如“的”、“是”、“在”等）。\n"
                "对于人物关键词应当包含职务或者职业属性。\n"
                "以逗号分隔关键词和同义词，关键词和同义词之间用分号分隔。\n"
                "---------------------\n"
                "示例：\n"
                "文本：爱丽丝是鲍勃的母亲。\n"
                "关键词：\n"
                "爱丽丝,母亲,鲍勃;妈妈\n"
                "文本：Philz是一家于1982年在伯克利创立的咖啡店。\n"
                "关键词：\n"
                "Philz,咖啡店,伯克利,1982;咖啡吧,咖啡馆\n"
                "---------------------\n",
                user_message="请从以下文本中提取关键词：{text}",
            ),
        ]
        super().__init__()
        self._prompts = {prompt.language: prompt for prompt in prompts}

    def get(self, language: str) -> KeywordsPromptModel:
        """Get the prompt for the specified language."""
        prompt = self._prompts.get(language)
        if not prompt:
            raise ValueError(f"No prompt found for language: {language}")
        system_message = SystemMessagePromptTemplate.from_template(prompt.system_message)
        human_message = HumanMessagePromptTemplate.from_template(prompt.user_message)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                system_message,
                human_message,
            ]
        )
        return prompt_template
