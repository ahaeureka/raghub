from typing import Any, Dict

from deeprag_core.operators.base_operator import BaseOperator
from deeprag_core.schemas.ner_model import NEROperatorOutputModel


class NEROperator(BaseOperator[NEROperatorOutputModel]):
    name = "NEROperator"
    description = "Operator for Named Entity Recognition"
    output_cls = NEROperatorOutputModel

    def post_process(self, output: Dict[str, Any]) -> Dict[str, Any]:
        content = output.get("content", {})
        if not content:
            output["named_entities"] = []
            return output
        output["named_entities"] = list(dict.fromkeys(output.get("content", {}).get("named_entities", [])))
        return output
