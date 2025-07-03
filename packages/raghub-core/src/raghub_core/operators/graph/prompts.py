from typing import List, Optional

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from raghub_core.prompts.base_prompt import BasePrompt
from raghub_core.schemas.graph_extract_model import GraphExtroactPromptModel


class GraphExtractorPrompt(BasePrompt):
    """
    Prompt for extracting a graph from a text.
    """

    def __init__(self, prompts: Optional[List[GraphExtroactPromptModel]] = None):
        super().__init__()
        prompts = prompts or [
            GraphExtroactPromptModel(
                language="zh",
                system_message="## 角色\n"
                "你是一个知识图谱工程专家，非常擅长从文本中精确抽取知识图谱的实体"
                "（主体、客体）和关系，并能对实体和关系的含义做出恰当的总结性描述。\n"
                "\n"
                "## 技能\n"
                "### 技能 1: 实体抽取\n"
                "--请按照如下步骤抽取实体--\n"
                "1. 准确地识别文本中的实体信息，一般是名词、代词等。\n"
                "2. 准确地识别实体的修饰性描述，一般作为定语对实体特征做补充。\n"
                "3. 对相同概念的实体（同义词、别称、代指），请合并为单一简洁的实体名，"
                "并合并它们的描述信息。\n"
                "4. 对合并后的实体描述信息做简洁、恰当、连贯的总结。\n"
                "\n"
                "### 技能 2: 关系抽取\n"
                "--请按照如下步骤抽取关系--\n"
                "1. 准确地识别文本中实体之间的关联信息，一般是动词、代词等。\n"
                "2. 准确地识别关系的修饰性描述，一般作为状语对关系特征做补充。\n"
                "3. 对相同概念的关系（同义词、别称、代指），请合并为单一简洁的关系名，"
                "并合并它们的描述信息。\n"
                "4. 对合并后的关系描述信息做简洁、恰当、连贯的总结。\n"
                "\n"
                "### 技能 3: 关联上下文\n"
                "- 关联上下文来自与当前待抽取文本相关的前置段落内容，"
                "可以为知识抽取提供信息补充。\n"
                "- 合理利用提供的上下文信息，知识抽取过程中出现的内容引用可能来自关联上下文。\n"
                "- 不要对关联上下文的内容做知识抽取，而仅作为关联信息参考。\n"
                "- 关联上下文是可选信息，可能为空。\n"
                "\n"
                "## 约束条件\n"
                "- 如果文本已提供了图结构格式的数据，直接转换为输出格式返回，"
                "不要修改实体或ID名称。"
                "- 尽可能多的生成文本中提及的实体和关系信息，但不要随意创造不存在的实体和关系。\n"
                "- 确保以第三人称书写，从客观角度描述实体名称、关系名称，以及他们的总结性描述。\n"
                "- 尽可能多地使用关联上下文中的信息丰富实体和关系的内容，这非常重要。\n"
                "- 如果实体或关系的总结描述为空，不提供总结描述信息，不要生成无关的描述信息。\n"
                "- 如果提供的描述信息相互矛盾，请解决矛盾并提供一个单一、连贯的描述。\n"
                "- 实体和关系的名称或者描述文本出现#和:字符时，使用_字符替换，其他字符不要修改。"
                "- 避免使用停用词和过于常见的词汇。\n"
                "- 对于人物实体，应当包含职务或者职业属性。\n"
                "- 不要使用任何非文本格式的输出，例如Markdown、HTML、LaTeX等。\n"
                "- Entities实体和Relationships关系在数量和内容上要对齐，所有的实体都应该在关系中出现，所有的关系中的来源实体名和目标实体名都应该在Entities中出现。\n"  # noqa: E501
                "\n"
                "## 输出格式\n"
                "Entities:\n"
                "(实体名#实体总结)\n"
                "...\n\n"
                "Relationships:\n"
                "(来源实体名#关系名#目标实体名#关系总结)\n"
                "...\n"
                "\n"
                "## 参考案例"
                "--案例仅帮助你理解提示词的输入和输出格式，请不要在答案中使用它们。--\n"
                "输入:\n"
                "```\n"
                "[上下文]:\n"
                "{context}\n"
                "..."
                "\n"
                "[文本]:\n"
                "{passage}"
                "```\n"
                "\n"
                "输出:\n"
                "{example_output}\n"
                "\n"
                "----\n"
                "\n",  # noqa: E501
                user_message="请根据接下来[上下文]提供的信息，按照上述要求，抽取[文本]中的实体和关系数据。\n"
                "\n"
                "[上下文]:\n"
                "{histories}\n"
                "\n"
                "[文本]:\n"
                "{text}\n"
                "\n"
                "[结果]:\n"
                "\n",
                context="Section 1:\n"
                "菲尔・贾伯的大儿子叫雅各布・贾伯。\n"
                "Section 2:\n"
                "菲尔・贾伯的小儿子叫比尔・贾伯。\n"
                "...",
                passage="Section 1:\n"
                "菲尔・贾伯的大儿子叫雅各布・贾伯。\n"
                "Section 2:\n"
                "菲尔・贾伯的小儿子叫比尔・贾伯。\n",
                example_output="```\n"
                "Entities:\n"
                "(菲尔・贾伯#菲尔兹咖啡创始人)\n"
                "(菲尔兹咖啡#加利福尼亚州伯克利创立的咖啡品牌)\n"
                "(雅各布・贾伯#菲尔・贾伯的大儿子)\n"
                "(美国多地#菲尔兹咖啡的扩展地区)\n"
                "\n"
                "Relationships:\n"
                "(菲尔・贾伯#创建#菲尔兹咖啡#1978年在加利福尼亚州伯克利创立)\n"
                "(菲尔兹咖啡#位于#加利福尼亚州伯克利#菲尔兹咖啡的创立地点)\n"
                "(菲尔・贾伯#拥有#雅各布・贾伯#菲尔・贾伯的大儿子)\n"
                "(雅各布・贾伯#管理#菲尔兹咖啡#在2005年担任首席执行官)\n"
                "(菲尔兹咖啡#扩展至#美国多地#菲尔兹咖啡的扩展范围)\n"
                "```\n",
            ),
            GraphExtroactPromptModel(
                language="en",
                system_message="## Role\n"
                "You are a knowledge graph engineering expert, highly skilled in accurately extracting entities\n "
                "(subjects and objects) and relationships from text to construct a knowledge graph.\n"
                " You are also adept at providing concise summary descriptions for the meanings of entities and relationships.\n"  # noqa: E501
                "\n"
                "## Skills\n"
                "### Skill 1: Entity Extraction\n"
                "--Follow these steps to extract entities--\n"
                "1. Accurately identify entity information in the text; usually nouns or pronouns.\n"
                "2. Accurately identify descriptive modifiers of entities; typically adjectives or clauses that provide additional characteristics of the entity.\n"  # noqa: E501
                "3. For entities representing the same concept (synonyms, alternative names, references), merge them into a single, concise entity name and combine their descriptive information.\n"  # noqa: E501
                "4. Summarize the combined entity descriptions concisely, appropriately, and coherently.\n"
                "\n"
                "### Skill 2: Relationship Extraction\n"
                "--Follow these steps to extract relationships--\n"
                "1. Accurately identify relational information between entities in the text; usually verbs or pronouns.\n"  # noqa: E501
                "2. Accurately identify descriptive modifiers of relationships; typically adverbs or clauses that add context to the relationship.\n"  # noqa: E501
                "3. For relationships representing the same concept (synonyms, alternative names, references), merge them into a single, concise relationship name and combine their descriptive information.\n"  # noqa: E501
                "4. Summarize the combined relationship descriptions concisely, appropriately, and coherently.\n"
                "\n"
                "### Skill 3: Context Association\n"
                "- Associated context comes from preceding paragraphs related to the current text being processed, which can provide supplementary information for knowledge extraction.\n"  # noqa: E501
                "- Reasonably utilize the provided associated context, as content references during knowledge extraction may originate from this context.\n"  # noqa: E501
                "- Do not perform knowledge extraction on the associated context itself but only use it as reference information.\n"  # noqa: E501
                "- Associated context is optional and may be empty.\n"
                "\n"
                "## Constraints\n"
                "- If the text already provides data in a graph structure format, directly convert it into the output format and return it without modifying entity or ID names.\n"  # noqa: E501
                "- Generate as much entity and relationship information as mentioned in the text, but do not invent non-existent entities or relationships.\n"  # noqa: E501
                "- Ensure writing is in the third person, objectively describing entity names, relationship names, and their summary descriptions.\n"  # noqa: E501
                "- Use as much information from the associated context as possible to enrich the content of entities and relationships, which is very important.\n"  # noqa: E501
                "- If the summary description of an entity or relationship is empty, do not provide any summary description and avoid generating irrelevant information.\n"  # noqa: E501
                "- If the provided descriptions contradict each other, resolve the contradictions and provide a single, coherent description.\n"  # noqa: E501
                "- When '#' or ':' characters appear in entity or relationship names or description texts, replace them with '_', do not modify other characters.\n"  # noqa: E501
                "- Avoid using stop words and overly common terms.\n"
                "- For person entities, include job titles or professional attributes.\n"
                "- Do not use any non-text output formats such as Markdown, HTML, LaTeX, etc.\n"
                "- Entities and Relationships must align in number and content; all entities should appear in relationships, and all source and target entity names in relationships should appear in Entities.\n"  # noqa: E501
                "\n"
                "## Output Format\n"
                "Entities:\n"
                "(Entity Name#Summary Description)\n"
                "...\n\n"
                "Relationships:\n"
                "(Source Entity Name#Relationship Name#Target Entity Name#Summary Description)\n"
                "...\n"
                "\n"
                "## Reference Example\n"
                "--Examples are only to help you understand the input and output format of the prompt; do not use them in your answer.--\n"  # noqa: E501
                "Input:\n"
                "```\n"
                "[Context]:\n"
                "{context}\n"
                "..."
                "\n"
                "[Text]:\n"
                "{passage}"
                "```\n"
                "\n"
                "Output:\n"
                "{example_output}\n"
                "\n"
                "----\n"
                "\n",  # noqa: E501
                user_message="Please extract the entities and relationships from [Text] according to the above requirements based on the information provided in [Context].\n"  # noqa: E501
                "\n"
                "[Context]:\n"
                "{histories}\n"
                "\n"
                "[Text]:\n"
                "{text}\n"
                "\n"
                "[Result]:\n"
                "\n",
                context="Section 1:\n"
                "Phil Jabber's eldest son is named Jacob Jabber.\n"
                "Section 2:\n"
                "Phil Jabber's youngest son is named Bill Jabber.\n"
                "...",
                passage="Section 1:\n"
                "The eldest son of Phil Jaber is named Jacob Jaber.\n"
                "Section 2:\n"
                "The youngest son of Phil Jaber is named Bill Jaber.\n",
                example_output="```\n"
                "Entities:\n"
                "(Phil Jaber#Founder of Phil's Coffee)\n"
                "(Phil's Coffee#Coffee brand established in Berkeley, California)\n"
                "(Jacob Jaber#Eldest son of Phil Jaber)\n"
                "(Various locations in the United States#Expansion areas of Phil's Coffee)\n"
                "\n"
                "Relationships:\n"
                "(Phil Jaber#Founded#Phil's Coffee#Established in 1978 in Berkeley, California)\n"
                "(Phil's Coffee#Located In#Berkeley, California#Location where Phil's Coffee was founded)\n"
                "(Phil Jaber#Has#Jacob Jaber#Phil Jaber's eldest son)\n"
                "(Jacob Jaber#Manages#Phil's Coffee#Served as CEO in 2005)\n"
                "(Phil's Coffee#Expanded To#Various locations in the United States#Expansion range of Phil's Coffee)\n"
                "```\n",
            ),
        ]
        # Default to an empty list if no prompts are provided
        self._prompts = {prompt.language: prompt for prompt in prompts}

    def get(self, lang="zh") -> ChatPromptTemplate:
        prompt = self._prompts.get(lang)
        if not prompt:
            raise ValueError(f"No prompt found for language: {lang}")
        # Create the prompt template
        system_message = SystemMessagePromptTemplate.from_template(prompt.system_message)
        human_message = HumanMessagePromptTemplate.from_template(prompt.user_message)

        # Build the few-shot prompt
        prompt_template = ChatPromptTemplate.from_messages(
            [
                system_message.format(
                    passage=prompt.passage, context=prompt.context, example_output=prompt.example_output
                ),
                human_message,
            ]
        )
        # logger.debug(f"NERPrompt: {prompt_template.messages}")
        return prompt_template
