from typing import List, Optional

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from raghub_core.prompts.base_prompt import BasePrompt
from raghub_core.schemas.prompt import PromptModel


class QueryIndentDetectionPrompt(BasePrompt):
    """
    A class to represent a query indent recursive prompt.

    This class is used to handle the recursive prompts for querying indents in a graph.
    It is designed to be used with the Raghub framework for graph operations.
    """

    def __init__(self, prompts: Optional[List[PromptModel]] = None):
        super().__init__()
        prompts = prompts or [
            PromptModel(
                language="en",
                system_message="""
A question is provided below. Given the question, analyze and classify it into one of the following categories:
1. SingleEntitySearch: search for the detail of the given entity.
2. OneHopEntitySearch: given one entity and one relation, search for all entities that have the relation with the given entity.
3. OneHopRelationSearch: given two entities, search for the relation between them.
4. TwoHopEntitySearch: given one entity and one relation, break that relation into two consecutive relation, then search all entities that have the two hop relation with the given entity.
5. FreestyleQuestion: questions that are not in above four categories. Search all related entities and two-hop sub-graphs centered on them.
Also return entities and relations that might be used for query generation in json format. Here are some examples to guide your classification:
---------------------
Example:
Question: Introduce TuGraph.
Return:
{{"category": "SingleEntitySearch", entities": ["TuGraph"], "relations": []}}
Question: Who commits code to TuGraph.
Return:
{{"category": "OneHopEntitySearch", "entities": ["TuGraph"], "relations": ["commit"]}}
Question: What is the relation between Alex and TuGraph?
Return:
{{"category": "OneHopRelationSearch", "entities": ["Alex", "TuGraph"], "relations": []}}
Question: Who is the colleague of Bob?
Return:
{{"category": "TwoHopEntitySearch", "entities": ["Bob"], "relations": ["work for"]}}
Question: Introduce TuGraph and DB-GPT separately.
Return:
{{"category": "FreestyleQuestion", "entities": ["TuGraph", "DBGPT"], "relations": []}}
---------------------

""",  # noqa: E501
                user_message="""
Text: {text}
Return:
""",
            ),
            PromptModel(
                language="zh",
                system_message="""
给定一个问题，请分析并归类到以下类别之一：
1. SingleEntitySearch：搜索给定实体的详细信息
2. OneHopEntitySearch：给定一个实体和一个关系，搜索与该实体存在该关系的所有实体
3. OneHopRelationSearch：给定两个实体，搜索它们之间的关系
4. TwoHopEntitySearch：给定一个实体和一个关系，将该关系拆分为两个连续关系，搜索与给定实体存在两跳关系的所有实体
5. FreestyleQuestion：不属于以上四类的问题。搜索所有相关实体及以其为中心的两跳子图
同时以JSON格式返回可能用于查询生成的实体和关系。参考示例如下：
---------------------
示例：
问题：介绍TuGraph
返回：
{{"category": "SingleEntitySearch", "entities": ["TuGraph"], "relations": []}}
问题：谁向TuGraph提交了代码
返回：
{{"category": "OneHopEntitySearch", "entities": ["TuGraph"], "relations": ["提交"]}}
问题：Alex和TuGraph之间是什么关系
返回：
{{"category": "OneHopRelationSearch", "entities": ["Alex", "TuGraph"], "relations": []}}
问题：Bob的同事是谁
返回：
{{"category": "TwoHopEntitySearch", "entities": ["Bob"], "relations": ["任职于"]}}
问题：分别介绍TuGraph和DB-GPT
返回：
{{"category": "FreestyleQuestion", "entities": ["TuGraph", "DBGPT"], "relations": []}}
---------------------
""",
                user_message="""
文本：{text}
返回：
""",
            ),
        ]
        self._prompts = {prompt.language: prompt for prompt in prompts}

    def get(self, language: str) -> PromptModel:
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
