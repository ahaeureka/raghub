"""
Build the prompt template for the Resource Description Framework (RDF) \
      generation task using the provided REPromptModel.
"""

import re
from typing import Any, Dict, List

from deeprag_core.operators.base_operator import BaseOperator
from deeprag_core.schemas.re_model import RDFOperatorOutputModel


class RDFOperator(BaseOperator[RDFOperatorOutputModel]):
    name = "RDFOperator"
    description = "Operator for generating RDF from text"
    output_cls = RDFOperatorOutputModel  # Assuming there's no specific output model defined yet

    def post_process(self, output: Dict[str, Any]) -> Dict[str, Any]:
        content = output.get("content", {})
        if not content:
            output["triples"] = []
            return output
        output["triples"] = self._text_processing(self._filter_invalid_triples(output["content"]["triples"]))
        return output

    def _filter_invalid_triples(self, triples: List[List[str]]) -> List[List[str]]:
        """
        Filters out invalid and duplicate triples from a list of triples.

        A valid triple meets the following criteria:
        1. It contains exactly three elements.
        2. It is unique within the list (no duplicates in the output).

        The function ensures:
        - Each valid triple is converted to a list of strings.
        - The order of unique, valid triples is preserved.
        - Do not apply any text preprocessing techniques or rules within this function.

        Args:
            triples (List[List[str]]):
                A list of triples (each a list of strings or elements that can be converted to strings).

        Returns:
            List[List[str]]:
                A list of unique, valid triples, each represented as a list of strings.
        """
        unique_triples = set()
        valid_triples = []

        for triple in triples:
            if len(triple) != 3:
                continue  # Skip triples that do not have exactly 3 elements

            valid_triple = [str(item) for item in triple]
            if tuple(valid_triple) not in unique_triples:
                unique_triples.add(tuple(valid_triple))
                valid_triples.append(valid_triple)

        return valid_triples

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
            print(f"Processed text_processing triple: {processed_triple}")
            processed_triples.append(processed_triple)
        return processed_triples
