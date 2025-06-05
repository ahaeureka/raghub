from raghub_core.operators.base_operator import BaseOperator
from raghub_core.schemas.graph_model import QueryIndentationModel


class QueryIndentDetectionOperator(BaseOperator[QueryIndentationModel]):
    """Operator for detecting query indentation in text."""

    name = "QueryIndentDetectionOperator"
    description = "Operator for detecting query indentation in text"
    output_cls = QueryIndentationModel
