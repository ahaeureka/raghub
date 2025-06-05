from typing import List, Optional

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from raghub_core.prompts.base_prompt import BasePrompt
from raghub_core.schemas.triple_model import TriplePromptModel


class TriplePrompts(BasePrompt):
    """Prompt for extracting keywords from text."""

    name = "TriplePrompts"
    description = "Prompt for extracting keywords from text"

    def __init__(self, prompts: Optional[List[TriplePromptModel]] = None):
        prompts = prompts or [
            TriplePromptModel(
                language="zh",
                system_message="你是一个专业的信息提取模型，任务是从用户问题中精准识别核心实体及其关系谓词，"
                "并构建结构化三元组（subject, predicate, object）。请严格遵循以下规则：\n"
                "1. **提取原则**：\n"
                "- 主语（subject）：问题中的核心实体（名词/专有名词）\n"
                "- 谓词（predicate）：描述实体属性或动作的关键关系词（动词/属性短语）,可能包含多个\n"
                "- 宾语（object）：\n"
                "  - 若问题明确给出属性值/对象则提取\n"
                '  - 若问题为询问句（如"是什么"），宾语留空\n'
                "  - 若问题隐含动作对象，需显式补全\n\n"
                "2. **处理规范**：\n"
                '- 精简实体：删除冗余修饰词（如"的"、"哪些"）\n'
                '- 谓词标准化：将疑问词转为属性描述（如"治愈属性"而非"有什么属性"）\n'
                "- 复合问题：提取每一个问题的三元组并给出数组json格式的输出\n\n"
                "3. **输出格式**：\n"
                "```json\n"
                "[{{\n"
                '"triple": {{\n'
                '"subject": "提取的主语",\n'
                '"predicate": ["提取的谓词"],\n'
                '"object": "提取的宾语（可为空）"\n'
                "}},\n"
                '"analysis": "处理逻辑说明（30字内）"\n'
                "}}]\n"
                "```\n\n"
                "----------------------------------------\n"
                "示例：\n"
                '- 问题："中国的首都是哪里？"\n'
                "输出：\n"
                "```json\n"
                "[{{\n"
                '"triple": {{\n'
                '"subject": "中国",\n'
                '"predicate": ["首都"],\n'
                '"object": []\n'
                "}},\n"
                '"analysis": "提取了中国的首都关系，宾语为空"\n'
                "}}]\n"
                "```\n\n"
                '- 问题："乔布斯创立了哪家公司"\n'
                "输出：\n"
                "```json\n"
                "[{{\n"
                '"triple": {{\n'
                '"subject": "乔布斯",\n'
                '"predicate": ["创立"],\n'
                '"object": []\n'
                "}},\n"
                '"analysis": "疑问句宾语缺失"\n'
                "}}]\n"
                "```\n\n"
                '- 问题:"洋甘菊能缓解失眠和焦虑"\n'
                "输出：\n"
                "```json\n"
                "[{{\n"
                '"triple": {{\n'
                '"subject": "洋甘菊",\n'
                '"predicate": ["缓解"],\n'
                '"object": ["失眠","焦虑"]\n'
                "}},\n"
                '"analysis": "提取洋甘菊的缓解作用，宾语为失眠和焦虑"\n'
                "}}]\n"
                "```\n\n"
                "- 问题：张伟参与的所有项目及合作伙伴\n"
                "输出：\n"
                "```json\n"
                "[{{\n"
                '"triple": {{\n'
                '"subject": "张伟",\n'
                '"predicate": ["参与","合作"],\n'
                '"object": []\n'
                "}}],\n",
                user_message="请从以下文本中提取实体-谓词三元组：{text}",
            ),
            TriplePromptModel(
                language="en",
                system_message="You are a professional information extraction model."
                "Your task is to precisely identify core entities and their relational predicates from user questions,"
                "then construct structured triples (subject, predicate, object). Strictly follow these rules:\n"
                "1. **Extraction Principles**:\n"
                "- Subject (subject): Core entities in the question (nouns/proper nouns)\n"
                "- Predicate (predicate): Key relational words describing entity attributes or "
                "actions (verbs/attribute phrases),"
                "may contain multiple entries\n"
                "- Object (object):\n"
                "  - Extract if explicit attribute values/objects are given\n"
                "  - Leave blank for interrogative questions (e.g., 'what is...')\n"
                "  - Explicitly complete implied action objects\n\n"
                "2. **Processing Standards**:\n"
                "- Simplify entities: Remove redundant modifiers (e.g., '的', '哪些')\n"
                "- Standardize predicates: Convert interrogatives to attribute descriptions "
                "(e.g., 'cure property' instead of 'has what property')\n"
                "- Compound questions: Extract triples for each sub-question and provide JSON array output\n\n"
                "3. **Output Format**:\n"
                "```json\n"
                "[{{\n"
                '"triple": {{\n'
                '"subject": "extracted subject",\n'
                '"predicate": ["extracted predicate"],\n'
                '"object": ["extracted object (may be empty)"]\n'
                "}},\n"
                '"analysis": "processing logic explanation (≤30 characters)"\n'
                "}}]\n"
                "```\n\n"
                "----------------------------------------\n"
                "Examples:\n"
                '- Question: "What is the capital of China?"\n'
                "Output:\n"
                "```json\n"
                "[{{\n"
                '"triple": {{\n'
                '"subject": "China",\n'
                '"predicate": ["capital"],\n'
                '"object": []\n'
                "}},\n"
                '"analysis": "Extracted capital relationship, object empty"\n'
                "}}]\n"
                "```\n\n"
                '- Question: "Which company did Steve Jobs found?"\n'
                "Output:\n"
                "```json\n"
                "[{{\n"
                '"triple": {{\n'
                '"subject": "Steve Jobs",\n'
                '"predicate": ["found"],\n'
                '"object": []\n'
                "}},\n"
                '"analysis": "Interrogative sentence missing object"\n'
                "}}]\n"
                "```\n\n"
                '- Question: "Chamomile can relieve insomnia and anxiety"\n'
                "Output:\n"
                "```json\n"
                "[{{\n"
                '"triple": {{\n'
                '"subject": "Chamomile",\n'
                '"predicate": ["relieve"],\n'
                '"object": ["insomnia", "anxiety"]\n'
                "}},\n"
                '"analysis": "Extracted chamomile\'s effects, objects: insomnia & anxiety"\n'
                "}}]\n"
                "```\n\n"
                '- Question: "All projects Zhang Wei participated in and partners"\n'
                "Output:\n"
                "```json\n"
                "[{{\n"
                '"triple": {{\n'
                '"subject": "Zhang Wei",\n'
                '"predicate": ["participate in", "collaborate with"],\n'
                '"object": []\n'
                "}},\n"
                '"analysis": "Extracted participation/collaboration relationships"\n'
                "}}]\n"
                "```\n",
                user_message="Please extract entity-predicate triples from the following text: {text}",
            ),
        ]
        super().__init__(prompts=prompts)
        self._prompts = {prompt.language: prompt for prompt in prompts}

    def get(self, language: str) -> TriplePromptModel:
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
