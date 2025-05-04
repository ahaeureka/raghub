from typing import List, Optional

from deeprag_core.prompts.base_prompt import BasePrompt
from deeprag_core.schemas.re_model import REPromptModel
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate


class REPrompt(BasePrompt):
    def __init__(self, prompts: Optional[List[REPromptModel]] = None):
        super().__init__()
        prompts = prompts or [
            REPromptModel(
                language="en",
                system_message="""Your task is to construct an \
                    RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.""",
                passage="""
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs. Radio City recently forayed into New Media 
in May 2008 with the launch of a music portal - PlanetRadiocity.com 
that offers music related news, videos, songs, and other music-related features.
""",
                named_entity_json="""
{{"named_entities": ["Radio City", "India", "3 July 2001", "Hindi", 
"English", "May 2008", "PlanetRadiocity.com"]}}
""",
                example_output="""
{{"triples": [
    ["Radio City", "located in", "India"],
    ["Radio City", "is", "private FM radio station"],
    ["Radio City", "started on", "3 July 2001"],
    ["Radio City", "plays songs in", "Hindi"],
    ["Radio City", "plays songs in", "English"],
    ["Radio City", "forayed into", "New Media"],
    ["Radio City", "launched", "PlanetRadiocity.com"],
    ["PlanetRadiocity.com", "launched in", "May 2008"],
    ["PlanetRadiocity.com", "is", "music portal"],
    ["PlanetRadiocity.com", "offers", "news"],
    ["PlanetRadiocity.com", "offers", "videos"],
    ["PlanetRadiocity.com", "offers", "songs"]
]}}
""",
            ),
            REPromptModel(
                language="zh",
                system_message="""你的任务是根据给定的段落和命名实体列表构建RDF(资源描述框架)图。
用JSON列表格式返回三元组，每个三元组代表RDF图中的一种关系。

请注意以下要求:
- 每个三元组应包含命名实体列表中至少一个(最好是两个)命名实体
- 明确将代词解析为特定名称以保持清晰""",
                passage="""
Radio City是印度第一家私营调频广播电台，成立于2001年7月3日。
它播放印地语、英语和地方歌曲。Radio City最近于2008年5月进军新媒体，
推出了音乐门户网站PlanetRadiocity.com，提供音乐相关的新闻、视频、歌曲和其他音乐相关功能。
""",
                named_entity_json="""
{{"named_entities": ["Radio City", "印度", "2001年7月3日", 
"印地语", "英语", "2008年5月", "PlanetRadiocity.com"]}}
""",
                example_output="""
{{"triples": [
    ["Radio City", "位于", "印度"],
    ["Radio City", "是", "私营调频广播电台"],
    ["Radio City", "成立于", "2001年7月3日"],
    ["Radio City", "播放", "印地语歌曲"],
    ["Radio City", "播放", "英语歌曲"],
    ["Radio City", "进军", "新媒体"],
    ["Radio City", "推出", "PlanetRadiocity.com"],
    ["PlanetRadiocity.com", "推出时间", "2008年5月"],
    ["PlanetRadiocity.com", "是", "音乐门户网站"],
    ["PlanetRadiocity.com", "提供", "新闻"],
    ["PlanetRadiocity.com", "提供", "视频"],
    ["PlanetRadiocity.com", "提供", "歌曲"]
]}}
""",
            ),
        ]
        self._prompts = {prompt.language: prompt for prompt in prompts}

    def get(self, lang="zh") -> ChatPromptTemplate:
        prompt = self._prompts.get(lang)
        if not prompt:
            raise ValueError(f"No prompt found for language: {lang}")

        system_message = SystemMessagePromptTemplate.from_template(prompt.system_message)

        # Create the few-shot examples
        example_input = f"""
Paragraph:
```
{prompt.passage}
```

{prompt.named_entity_json}
"""

        # Build the prompt template
        prompt_template = ChatPromptTemplate.from_messages(
            [
                system_message,
                ("user", example_input),
                ("assistant", prompt.example_output),
                (
                    "user",
                    """
Paragraph:
```
{passage}
```

{named_entity_json}
""",
                ),
            ]
        )

        return prompt_template
