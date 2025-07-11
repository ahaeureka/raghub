from typing import Any, Dict

from loguru import logger
from raghub_core.operators.base_operator import BaseOperator
from raghub_core.schemas.ner_model import NEROperatorOutputModel


class NEROperator(BaseOperator[NEROperatorOutputModel]):
    name = "NEROperator"
    description = "Operator for Named Entity Recognition"
    output_cls = NEROperatorOutputModel

    def post_process(self, output: Dict[str, Any]) -> Dict[str, Any]:
        content = output.get("content", {})
        logger.debug(f"NEROperator post_process content: {content}")
        if not content:
            output["named_entities"] = []
            return output
        if isinstance(content, list):
            output["named_entities"] = content
        elif isinstance(content, dict):
            output["named_entities"] = list(dict.fromkeys(output.get("content", {}).get("named_entities", [])))
        return output
