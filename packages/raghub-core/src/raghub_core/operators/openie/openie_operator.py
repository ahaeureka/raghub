import json

from raghub_core.operators.ner.ner_operator import NEROperator
from raghub_core.operators.openie.base_openIE import BaseOpenIE
from raghub_core.operators.rdf.rdf_operator import RDFOperator
from raghub_core.schemas.openie_mdoel import OpenIEModel


class OpenIEOperator(BaseOpenIE):
    name = "OpenIEOperator"

    def __init__(self, ner_operator: NEROperator, re_operator: RDFOperator):
        self._ner_operator = ner_operator
        self._re_operator = re_operator
        super().__init__()

    async def extract(self, text: str, lang: str = "zh") -> OpenIEModel:
        ner_output = await self._ner_operator.execute({"paragraph": text}, lang=lang)

        re_output = await self._re_operator.execute(
            {"passage": text, "named_entity_json": json.dumps(ner_output.model_dump(), ensure_ascii=False)},
            lang=lang,
        )

        # Combine the outputs from NER and RDF operators
        return OpenIEModel(
            ner=ner_output.named_entities,
            triples=[(subject, predicate, object_) for [subject, predicate, object_] in re_output.triples],
        )
