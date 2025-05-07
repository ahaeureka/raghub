from typing import List

from pydantic import Field
from raghub_core.schemas.operator_model import OperatorOutputModel
from raghub_core.schemas.prompt import PromptModel


class REPromptModel(PromptModel):
    language: str = Field(..., description="The language of the text.")
    passage: str = Field(..., description="The input passage/text to extract relations from.")
    system_message: str = Field(
        ..., description="The system message defining the relation extraction task requirements."
    )
    named_entity_json: str = Field(
        ..., description="The JSON string containing named entities identified in the passage."
    )
    example_output: str = Field(
        ..., description="An example output showing the extracted relations in RDF triple format."
    )


# {"triples": [
#     ["Radio City", "位于", "印度"],
#     ["Radio City", "是", "私营调频广播电台"],
#     ["Radio City", "成立于", "2001年7月3日"],
#     ["Radio City", "播放", "印地语歌曲"],
#     ["Radio City", "播放", "英语歌曲"],
#     ["Radio City", "进军", "新媒体"],
#     ["Radio City", "推出", "PlanetRadiocity.com"],
#     ["PlanetRadiocity.com", "推出时间", "2008年5月"],
#     ["PlanetRadiocity.com", "是", "音乐门户网站"],
#     ["PlanetRadiocity.com", "提供", "新闻"],
#     ["PlanetRadiocity.com", "提供", "视频"],
#     ["PlanetRadiocity.com", "提供", "歌曲"]
# ]}
class RDFOperatorOutputModel(OperatorOutputModel):
    triples: List[List[str]] = Field(..., description="List of triples extracted from the text.")
