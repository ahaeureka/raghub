from typing import List, Optional

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from raghub_core.prompts.base_prompt import BasePrompt
from raghub_core.schemas.ner_model import NERPromptModel


class NERPrompt(BasePrompt):
    def __init__(self, prompts: Optional[List[NERPromptModel]] = None):
        super().__init__()
        prompts = prompts or [
            NERPromptModel(
                language="en",
                system_message="Your task is to extract named entities from the given paragraph.\nRespond with a JSON list of entities.",  # noqa: E501
                user_message="""{paragraph}""",
                paragraph="""Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal\
      - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.""",
                example_output="""
                           {{"named_entities": ["Radio City", "India", "3 July 2001", "Hindi", 
                           "English", "May 2008", "PlanetRadiocity.com"]}}
                           """,
            ),
            NERPromptModel(
                language="zh",
                system_message="你的任务是从给定的段落中提取命名实体。\n用JSON列表格式返回实体。",
                paragraph="""
                Radio City \n
                           Radio City是印度第一家私营调频广播电台，成立于2001年7月3日。
                           它播放印地语、英语和地方歌曲。Radio City最近于2008年5月进军新媒体，
                           推出了音乐门户网站PlanetRadiocity.com，提供音乐相关的新闻、视频、歌曲和其他音乐相关功能。
                           """,
                example_output="""
                           {{"named_entities": ["Radio City", "印度", "2001年7月3日", 
                           "印地语", "英语", "2008年5月", "PlanetRadiocity.com"]}}
                           """,
                user_message="""{paragraph}""",
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
            [system_message, ("user", prompt.paragraph), ("assistant", prompt.example_output), human_message]
        )
        # logger.debug(f"NERPrompt: {prompt_template.messages}")
        return prompt_template
