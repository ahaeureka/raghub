import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from deeprag_core.operators.base_operator import BaseOperator
from deeprag_core.schemas.dspy_filter_model import DSPyFilterOutputModel
from langchain_core.messages import AIMessage
from loguru import logger
from pydantic import BaseModel, Field, TypeAdapter


class Fact(BaseModel):
    fact: list[list[str]] = Field(
        description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]"
    )


class DSPyFilter(BaseOperator[DSPyFilterOutputModel]):
    name = "DSPyFilter"
    output_cls = DSPyFilterOutputModel
    description = "DSPyFilter operator for filtering facts based on a given question."

    def output_parser(self, content: AIMessage) -> DSPyFilterOutputModel:
        sections: List[Tuple[Optional[str], List[str]]] = [(None, [])]
        response: str = content.content
        logger.debug(f"DSPyFilter response: {response}")
        field_header_pattern = re.compile("\\[\\[ ## (\\w+) ## \\]\\]")
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        result_sections = [(k, "\n".join(v).strip()) for k, v in sections]
        parsed = []
        for k, value in result_sections:
            if k == "fact_after_filter":
                try:
                    # fields[k] = parse_value(v, signature.output_fields[k].annotation) if _parse_values else v
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        try:
                            parsed_value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            parsed_value = value
                    parsed = TypeAdapter(Fact).validate_python(parsed_value).fact
                except Exception as e:
                    logger.error(
                        f"Error parsing field {k}: {e}.\n\n\t\tOn attempting to parse the value\n```\n{value}\n```"
                    )
        return DSPyFilterOutputModel(name="DSPyFilter", fact_after_filter=parsed, completed="[[ ## completed ## ]]")

    def post_process(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return output
