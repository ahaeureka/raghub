from typing import List, Optional

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from raghub_core.prompts.base_prompt import BasePrompt
from raghub_core.schemas.summarize_model import SummarizePromptModel


class CommunitySummarizerPrompts(BasePrompt):
    """
    This class contains the prompts used for summarizing communities.
    """

    def __init__(self, prompts: Optional[List[SummarizePromptModel]] = None):
        """
        Initializes the CommunitySummarizerPrompts class.

        Args:
            prompts (Optional[List[SummarizePromptModel]]): A list of prompt models for summarization.
        """
        prompts = prompts or [
            SummarizePromptModel(
                language="zh",
                system_message="## 角色\n"
                "你非常擅长知识图谱的信息总结，能根据给定的知识图谱中的实体和关系的名称以及描述"
                "信息，全面、恰当地对知识图谱子图信息做出总结性描述，并且不会丢失关键的信息。\n"
                "\n"
                "## 技能\n"
                "### 技能 1: 实体识别\n"
                "- 准确地识别[Entities:]章节中的实体信息，包括实体名、实体描述信息。\n"
                "- 实体信息的一般格式有:\n"
                "(实体名)\n"
                "(实体名:实体描述)\n"
                "(实体名:实体属性表)\n"
                "(文本块ID:文档块内容)\n"
                "(目录ID:目录名)\n"
                "(文档ID:文档名称)\n"
                "\n"
                "### 技能 2: 关系识别\n"
                "- 准确地识别[Relationships:]章节中的关系信息，包括来源实体名、关系名、"
                "目标实体名、关系描述信息，实体名也可能是文档ID、目录ID、文本块ID。\n"
                "- 关系信息的一般格式有:\n"
                "(来源实体名)-[关系名]->(目标实体名)\n"
                "(来源实体名)-[关系名:关系描述]->(目标实体名)\n"
                "(来源实体名)-[关系名:关系属性表]->(目标实体名)\n"
                "(文本块ID)-[包含]->(实体名)\n"
                "(目录ID)-[包含]->(文本块实体)\n"
                "(目录ID)-[包含]->(子目录ID)\n"
                "(文档ID)-[包含]->(文本块实体)\n"
                "(文档ID)-[包含]->(目录ID)\n"
                "\n"
                "### 技能 3: 图结构理解\n"
                "--请按照如下步骤理解图结构--\n"
                "1. 正确地将关系信息中的来源实体名与实体信息关联。\n"
                "2. 正确地将关系信息中的目标实体名与实体信息关联。\n"
                "3. 根据提供的关系信息还原出图结构。"
                "\n"
                "### 技能 4: 知识图谱总结\n"
                "--请按照如下步骤总结知识图谱--\n"
                "1. 确定知识图谱表达的主题或话题，突出关键实体和关系。"
                "2. 使用准确、恰当、简洁的语言总结图结构表达的信息，不要生成与图结构中无关的信息。"
                "\n"
                "## 约束条件\n"
                "- 不要在答案中描述你的思考过程，直接给出用户问题的答案，不要生成无关信息。\n"
                "- 确保以第三人称书写，从客观角度对知识图谱表达的信息进行总结性描述。\n"
                "- 如果实体或关系的描述信息为空，对最终的总结信息没有贡献，不要生成无关信息。\n"
                "- 如果提供的描述信息相互矛盾，请解决矛盾并提供一个单一、连贯的描述。\n"
                "- 避免使用停用词和过于常见的词汇。\n"
                "\n",
                input_example="## 参考案例\n"
                "--案例仅帮助你理解提示词的输入和输出格式，请不要在答案中使用它们。--\n"
                "输入:\n"
                "```\n"
                "Entities:\n"
                "(菲尔・贾伯#菲尔兹咖啡创始人)\n"
                "(菲尔兹咖啡#加利福尼亚州伯克利创立的咖啡品牌)\n"
                "(雅各布・贾伯#菲尔・贾伯的儿子)\n"
                "(美国多地#菲尔兹咖啡的扩展地区)\n"
                "\n"
                "Relationships:\n"
                "(菲尔・贾伯#创建#菲尔兹咖啡#1978年在加利福尼亚州伯克利创立)\n"
                "(菲尔兹咖啡#位于#加利福尼亚州伯克利#菲尔兹咖啡的创立地点)\n"
                "(菲尔・贾伯#拥有#雅各布・贾伯#菲尔・贾伯的儿子)\n"
                "(雅各布・贾伯#担任#首席执行官#在2005年成为菲尔兹咖啡的首席执行官)\n"
                "(菲尔兹咖啡#扩展至#美国多地#菲尔兹咖啡的扩展范围)\n"
                "```\n"
                "\n",
                output_example="输出:\n"
                "```\n"
                "Summary:"
                "菲尔兹咖啡是由菲尔・贾伯在1978年于加利福尼亚州伯克利创立的咖啡品牌。"
                "菲尔・贾伯的儿子雅各布・贾伯在2005年接任首席执行官，领导公司扩展到了美国多地。"
                "进一步巩固了菲尔兹咖啡作为加利福尼亚州伯克利创立的咖啡品牌的市场地位。\n"
                "\n"
                "Keywords:\n"
                "- 菲尔・贾伯\n"
                "- 菲尔兹咖啡\n"
                "- 加利福尼亚州伯克利\n"
                "- 雅各布・贾伯\n"
                "- 首席执行官\n"
                "```\n"
                "\n"
                "----\n"
                "\n",
                user_message="请根据接下来[知识图谱]提供的信息，按照上述要求，总结知识图谱表达的信息。\n"
                "\n"
                "[知识图谱]:\n"
                "{graph}\n"
                "\n"
                "[总结]:\n"
                "\n",
            ),
            SummarizePromptModel(
                language="en",
                system_message="## Role\n"
                "You are highly skilled in knowledge graph summarization, capable of providing comprehensive and appropriate descriptive summaries of subgraph information from knowledge graphs based on entity names, relationship names, and their descriptions. Critical information will never be omitted.\n"  # noqa: E501
                "\n"
                "## Skills\n"
                "### Skill 1: Entity Recognition\n"
                "- Accurately identify entity information in the [Entities:] section including entity names and descriptions.\n"  # noqa: E501
                "- Common entity formats include:\n"
                "(Entity Name)\n"
                "(Entity Name:Entity Description)\n"
                "(Entity Name:Entity Attribute Table)\n"
                "(Text Block ID:Document Block Content)\n"
                "(Directory ID:Directory Name)\n"
                "(Document ID:Document Title)\n"
                "\n"
                "### Skill 2: Relationship Recognition\n"
                "- Accurately identify relationship information in the [Relationships:] section including source entity name, relationship name, target entity name, and relationship descriptions. Entity names may also be Document IDs, Directory IDs, or Text Block IDs.\n"  # noqa: E501
                "- Common relationship formats include:\n"
                "(Source Entity Name)-[Relationship Name]->(Target Entity Name)\n"
                "(Source Entity Name)-[Relationship Name:Relationship Description]->(Target Entity Name)\n"
                "(Source Entity Name)-[Relationship Name:Relationship Attribute Table]->(Target Entity Name)\n"
                "(Text Block ID)-[Contains]->(Entity Name)\n"
                "(Directory ID)-[Contains]->(Text Block Entity)\n"
                "(Directory ID)-[Contains]->(Subdirectory ID)\n"
                "(Document ID)-[Contains]->(Text Block Entity)\n"
                "(Document ID)-[Contains]->(Directory ID)\n"
                "\n"
                "### Skill 3: Graph Structure Understanding\n"
                "--Follow these steps to understand the graph structure--\n"
                "1. Correctly associate source entity names in relationships with entity information.\n"
                "2. Correctly associate target entity names in relationships with entity information.\n"
                "3. Reconstruct the graph structure based on provided relationships.\n"
                "\n"
                "### Skill 4: Knowledge Graph Summarization\n"
                "--Follow these steps to summarize the knowledge graph--\n"
                "1. Identify the main topic or theme expressed in the knowledge graph, emphasizing key entities and relationships.\n"  # noqa: E501
                "2. Use accurate, appropriate, and concise language to summarize the information conveyed by the graph structure without generating irrelevant content.\n"  # noqa: E501
                "\n"
                "## Constraints\n"
                "- Do not describe your thought process in the answer. Directly provide the answer to the user's question without generating irrelevant information.\n"  # noqa: E501
                "- Ensure the summary is written in third person from an objective perspective regarding the knowledge graph's content.\n"  # noqa: E501
                "- If entity or relationship descriptions are empty, they should not contribute to the final summary. Avoid generating irrelevant information.\n"  # noqa: E501
                "- If provided descriptions conflict, resolve contradictions and provide a single coherent description.\n"  # noqa: E501
                "- Avoid using stop words and overly common terms.\n"
                "\n",
                input_example="## Reference Example\n"
                "--This example is for understanding input/output formats only. Do not use it in your answers.--\n"
                "Input:\n"
                "```\n"
                "Entities:\n"
                "(Phil Jacobs # Founder of Peet's Coffee)\n"
                "(Peet's Coffee # Coffee brand established in Berkeley, California)\n"
                "(Jacob Jacobs # Son of Phil Jacobs)\n"
                "(Multiple U.S. locations # Expansion areas of Peet's Coffee)\n"
                "\n"
                "Relationships:\n"
                "(Phil Jacobs # created # Peet's Coffee # Founded in Berkeley, California in 1978)\n"
                "(Peet's Coffee # located in # Berkeley, California # Founding location of Peet's Coffee)\n"
                "(Phil Jacobs # owns # Jacob Jacobs # Phil Jacobs' son)\n"
                "(Jacob Jacobs # serves as # CEO # Became CEO of Peet's Coffee in 2005)\n"
                "(Peet's Coffee # expanded to # Multiple U.S. locations # Expansion scope of Peet's Coffee)\n"
                "```\n"
                "\n",
                output_example="Output:\n"
                "```\n"
                "Summary:"
                "Peet's Coffee was established in 1978 in Berkeley, California by Phil Jacobs. His son Jacob Jacobs became CEO in 2005, leading the company's expansion across multiple U.S. locations and strengthening its market position as a coffee brand originating from Berkeley, California.\n"  # noqa: E501
                "\n"
                "Keywords:\n"
                "- Phil Jacobs\n"
                "- Peet's Coffee\n"
                "- Berkeley, California\n"
                "- Jacob Jacobs\n"
                "- CEO\n"
                "```\n"
                "\n"
                "----\n"
                "\n",
                user_message="Based on the [Knowledge Graph] information provided below, please generate a summary following the requirements outlined above.\n"  # noqa: E501
                "\n"
                "[Knowledge Graph]:\n"
                "{graph}\n"
                "\n"
                "[Summary]:\n"
                "\n",
            ),
        ]
        super().__init__()
        self._prompts = {prompt.language: prompt for prompt in prompts}

    def get(self, lang: str) -> SummarizePromptModel:
        """
        Retrieves the prompt model for the specified language.

        Args:
            language (str): The language code for the desired prompt.

        Returns:
            SummarizePromptModel: The prompt model for the specified language.
        """
        prompt = self._prompts.get(lang)
        if not prompt:
            raise ValueError(f"No prompt found for language: {lang}")
        # Create the prompt template
        system_message = SystemMessagePromptTemplate.from_template(
            prompt.system_message + prompt.input_example + prompt.output_example
        )
        human_message = HumanMessagePromptTemplate.from_template(prompt.user_message)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                system_message,
                human_message,
            ]
        )
        return prompt_template

    # def output_parser(self, output: str) -> Dict[str, List[str]]:
    #     """
    #     Parses the output from the model into a structured format.

    #     Args:
    #         output (str): The output string from the model.

    #     Returns:
    #         Dict[str, List[str]]: A dictionary containing the summary and keywords.
    #     """
    #     summary_text = ""
    #     keywords = []
    #     lines = output.split("\n")
    #     for line in lines:
    #         line = line.strip()
    #         if line.startswith("Summary:"):
    #             summary_text = line[len("Summary:"):].strip()
    #         elif line.startswith("-"):
    #             keywords.append(line[1:].strip())
    #     return {
    #         "summary": summary_text,
    #         "keywords": keywords,
    #     }
