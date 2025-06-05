"""
Build the prompt template for the Resource Description Framework (Triple) \
      generation task using the provided REPromptModel.
"""

import re
from typing import Any, Dict, List

from raghub_core.operators.base_operator import BaseOperator
from raghub_core.schemas.triple_model import TriplesOperatorOutputModel


class TripleOperator(BaseOperator[TriplesOperatorOutputModel]):
    name = "TripleOperator"
    description = "Operator for generating Triple from text"
    output_cls = TriplesOperatorOutputModel  # Assuming there's no specific output model defined yet

    def post_process(self, output: Dict[str, Any]) -> Dict[str, Any]:
        # content = output.get("content", {})
        # if not content:
        #     output["triple"] = {}
        #     return output
        # output["triple"] = self._text_processing(output["content"])
        return output

    def _text_processing(self, triples: List[List[str]]) -> List[List[str]]:
        """
        Apply text preprocessing to each element of the triples.

        This includes:
        - Converting all text to lowercase.
        - Removing all non-alphanumeric characters except spaces.
        - Stripping leading and trailing spaces.

        Args:
           triples (List[List[str]]):
                A list of triples (each a list of strings).

        Returns:
           List[List[str]]:
                A list of triples with each element processed as described.
        """
        processed_triples = []
        for triple in triples:
            processed_triple = [re.sub("[^A-Za-z0-9 ]", " ", item.lower()).strip() for item in triple]
            processed_triples.append(processed_triple)
        return processed_triples
