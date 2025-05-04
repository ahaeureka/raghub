import copy
import json
import os
from typing import Any, Dict, List, Optional

from deeprag_core.prompts.base_prompt import BasePrompt
from deeprag_core.schemas.dspy_filter_model import DSPyFilterPromptModel
from langchain.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from loguru import logger


def get_query_instruction(linking_method, lang="en"):
    en_instructions = {
        "ner_to_node": "Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.",
        "query_to_node": "Given a question, retrieve relevant phrases that are mentioned in this question.",
        "query_to_fact": "Given a question, retrieve relevant triplet facts that matches this question.",
        "query_to_sentence": "Given a question, retrieve relevant sentences that best answer the question.",
        "query_to_passage": "Given a question, retrieve relevant documents that best answer the question.",
    }
    zh_instructions = {
        "ner_to_node": "给定一个短语，检索与此短语最匹配的同义词或相关短语。",
        "query_to_node": "给定一个问题，检索该问题中提到的相关短语。",
        "query_to_fact": "给定一个问题，检索与此问题匹配的相关三元组事实。",
        "query_to_sentence": "给定一个问题，检索最能回答该问题的相关句子。",
        "query_to_passage": "给定一个问题，检索最能回答该问题的相关文档。",
    }
    instructions = en_instructions if lang == "en" else zh_instructions
    default_instruction = "Given a question, retrieve relevant documents that best answer the question."
    return instructions.get(linking_method, default_instruction)


def get_best_dspy_prompt(best_dspy_prompt_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns a dictionary containing the best prompt for the DSPy model.
    """
    if best_dspy_prompt_path and os.path.exists(best_dspy_prompt_path):
        with open(best_dspy_prompt_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {
        "prog": {
            "lm": None,
            "traces": [],
            "train": [],
            "demos": [
                {
                    "augmented": True,
                    "question": "Are Imperial River (Florida) and Amaradia (Dolj) both located in the same country?",
                    "fact_before_filter": '{{"fact": [["imperial river", "is located in", "florida"], ["imperial river", "is a river in", "united states"], ["imperial river", "may refer to", "south america"], ["amaradia", "flows through", "ro ia de amaradia"], ["imperial river", "may refer to", "united states"]]}}',  # noqa: E501
                    "fact_after_filter": '{{"fact":[["imperial river","is located in","florida"],["imperial river","is a river in","united states"],["amaradia","flows through","ro ia de amaradia"]]}}',  # noqa: E501
                },
                {
                    "augmented": True,
                    "question": "When is the director of film The Ancestor 's birthday?",
                    "fact_before_filter": '{{"fact": [["jean jacques annaud", "born on", "1 october 1943"], ["tsui hark", "born on", "15 february 1950"], ["pablo trapero", "born on", "4 october 1971"], ["the ancestor", "directed by", "guido brignone"], ["benh zeitlin", "born on", "october 14  1982"]]}}',  # noqa: E501
                    "fact_after_filter": '{{"fact":[["the ancestor","directed by","guido brignone"]]}}',
                },
                {
                    "augmented": True,
                    "question": "In what geographic region is the country where Teafuone is located?",
                    "fact_before_filter": '{{"fact": [["teafuaniua", "is on the", "east"], ["motuloa", "lies between", "teafuaniua"], ["motuloa", "lies between", "teafuanonu"], ["teafuone", "is", "islet"], ["teafuone", "located in", "nukufetau"]]}}',  # noqa: E501
                    "fact_after_filter": '{{"fact":[["teafuone","is","islet"],["teafuone","located in","nukufetau"]]}}',
                },
                {
                    "augmented": True,
                    "question": "When did the director of film S.O.B. (Film) die?",
                    "fact_before_filter": '{{"fact": [["allan dwan", "died on", "28 december 1981"], ["s o b", "written and directed by", "blake edwards"], ["robert aldrich", "died on", "december 5  1983"], ["robert siodmak", "died on", "10 march 1973"], ["bernardo bertolucci", "died on", "26 november 2018"]]}}',  # noqa: E501
                    "fact_after_filter": '{{"fact":[["s o b","written and directed by","blake edwards"]]}}',
                },
                {
                    "augmented": True,
                    "question": "Do both films: Gloria (1980 Film) and A New Life (Film) have the directors from the same country?",  # noqa: E501
                    "fact_before_filter": '{{"fact": [["sebasti n lelio watt", "received acclaim for directing", "gloria"], ["gloria", "is", "1980 american thriller crime drama film"], ["a brand new life", "is directed by", "ounie lecomte"], ["gloria", "written and directed by", "john cassavetes"], ["a new life", "directed by", "alan alda"]]}}',  # noqa: E501
                    "fact_after_filter": '{{"fact":[["gloria","is","1980 american thriller crime drama film"],["gloria","written and directed by","john cassavetes"],["a new life","directed by","alan alda"]]}}',  # noqa: E501
                },
                {
                    "augmented": True,
                    "question": "What is the date of death of the director of film The Old Guard (1960 Film)?",
                    "fact_before_filter": '{{"fact": [["the old guard", "is", "1960 french comedy film"], ["gilles grangier", "directed", "the old guard"], ["the old guard", "directed by", "gilles grangier"], ["the old fritz", "directed by", "gerhard lamprecht"], ["oswald albert mitchell", "directed", "old mother riley series of films"]]}}',  # noqa: E501
                    "fact_after_filter": '{{"fact":[["the old guard","is","1960 french comedy film"],["gilles grangier","directed","the old guard"],["the old guard","directed by","gilles grangier"]]}}',  # noqa: E501
                },
                {
                    "augmented": True,
                    "question": "When is the composer of film Aulad (1968 Film) 's birthday?",
                    "fact_before_filter": '{{"fact": [["aulad", "has music composed by", "chitragupta shrivastava"], ["aadmi sadak ka", "has music by", "ravi"], ["ravi shankar sharma", "composed music for", "hindi films"], ["gulzar", "was born on", "18 august 1934"], ["aulad", "is a", "1968 hindi language drama film"]]}}',  # noqa: E501
                    "fact_after_filter": '{{"fact":[["aulad","has music composed by","chitragupta shrivastava"],["aulad","is a","1968 hindi language drama film"]]}}',  # noqa: E501
                },
                {
                    "question": "How many households were in the city where Angelical Tears located?",
                    "fact_before_filter": '{{"fact": [["dow city", "had", "219 households"], ["tucson", "had", "229 762 households"], ["atlantic city", "has", "15 504 households"], ["angelical tears", "located in", "oklahoma city"], ["atlantic city", "had", "15 848 households"]]}}',  # noqa: E501
                    "fact_after_filter": '{{"fact": [["angelical tears", "located in", "oklahoma city"]]}}',
                },
                {
                    "question": "Did the movies In The Pope'S Eye and Virgin Mountain, originate from the same country?",  # noqa: E501
                    "fact_before_filter": '{{"fact": [["virgin mountain", "released in", "icelandic cinemas"], ["virgin mountain", "directed by", "dagur k ri"], ["virgin mountain", "icelandic title is", "f si"], ["virgin mountain", "won", "2015 nordic council film prize"], ["virgin mountain", "is a", "2015 icelandic drama film"]]}}',  # noqa: E501
                    "fact_after_filter": '{{"fact": [["virgin mountain", "released in", "icelandic cinemas"], ["virgin mountain", "directed by", "dagur k ri"], ["virgin mountain", "icelandic title is", "f si"], ["virgin mountain", "won", "2015 nordic council film prize"], ["virgin mountain", "is a", "2015 icelandic drama film"]]}}',  # noqa: E501
                },
                {
                    "question": "Which film has the director who died earlier, The Virtuous Model or Bulldog Drummond'S Peril?",  # noqa: E501
                    "fact_before_filter": '{{"fact": [["the virtuous model", "is", "1919 american silent drama film"], ["bulldog drummond s peril", "directed by", "james p  hogan"], ["the virtuous model", "directed by", "albert capellani"], ["bulldog drummond s revenge", "directed by", "louis king"], ["bulldog drummond s peril", "is", "american film"]]}}',  # noqa: E501
                    "fact_after_filter": '{{"fact": [["the virtuous model", "is", "1919 american silent drama film"], ["bulldog drummond s peril", "directed by", "james p  hogan"], ["the virtuous model", "directed by", "albert capellani"], ["bulldog drummond s peril", "is", "american film"]]}}',  # noqa: E501
                },
            ],
            "signature": {
                "instructions": 'You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. You must select up to 4 relevant facts from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer. The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts are relevant, return an empty list, {"fact": []}. The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. The future of critical decision-making relies on your ability to accurately filter and present relevant information.',  # noqa: E501
                "fields": [
                    {"prefix": "Question:", "description": "Query for retrieval"},
                    {"prefix": "Fact Before Filter:", "description": "Candidate facts to be filtered"},
                    {"prefix": "Fact After Filter:", "description": "Filtered facts in JSON format"},
                ],
            },
            "system": 'Your input fields are:\n1. `question` (str): Query for retrieval\n2. `fact_before_filter` (str): Candidate facts to be filtered\n\nYour output fields are:\n1. `fact_after_filter` (Fact): Filtered facts in JSON format\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{{question}}\n\n[[ ## fact_before_filter ## ]]\n{{fact_before_filter}}\n\n[[ ## fact_after_filter ## ]]\n{{fact_after_filter}}        # note: the value you produce must be pareseable according to the following JSON schema: {{"type": "object", "properties": {{"fact": {{"type": "array", "description": "A list of facts, each fact is a list of 3 strings: [subject, predicate, object]", "items": {{"type": "array", "items": {{"type": "string"}}, "title": "Fact"}}, "required": ["fact"], "title": "Fact"}}\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is: \n        You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. You must select up to 4 relevant facts from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer. The output should be in JSON format, e.g., {{"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}}, and if no facts are relevant, return an empty list, {{"fact": []}}. The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. The future of critical decision-making relies on your ability to accurately filter and present relevant information.',  # noqa: E501
        }
    }


class DSPyRerankPrompt(BasePrompt):
    def __init__(self, prompts: Optional[List[DSPyFilterPromptModel]] = None, best_dspy_path=None):
        super().__init__()
        dspy_file_path = best_dspy_path
        self.dspy_saved = get_best_dspy_prompt(
            dspy_file_path
        )  # Assuming best_dspy_prompt is defined elsewhere in the code or imported from another module.
        # If not, you need to define it or load it from a file.
        system_prompt = self.dspy_saved["prog"]["system"]

        prompts = prompts or [
            DSPyFilterPromptModel(
                system_message=system_prompt,
                user_message="""[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n[[ ## completed ## ]]""",  # noqa: E501
                example_output="""[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]""",
                language="en",
            ),
            DSPyFilterPromptModel(
                language="zh",
                system_message=system_prompt,
                user_message="""[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n[[ ## completed ## ]]""",  # noqa: E501
                example_output="""[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]""",
            ),
        ]
        self._prompts = {prompt.language: prompt for prompt in prompts}

    def get(self, lang="zh") -> ChatPromptTemplate:
        prompt = self._prompts.get(lang)
        if not prompt:
            raise ValueError(f"No prompt found for language: {lang}")
        system_message = SystemMessagePromptTemplate.from_template(prompt.system_message)
        human_message = HumanMessagePromptTemplate.from_template(prompt.user_message)
        ai_message = AIMessagePromptTemplate.from_template(
            prompt.example_output
        )  # Assuming you have an AI message template
        messages = []
        for demo in self.dspy_saved["prog"]["demos"]:
            messages.append(
                (
                    "human",
                    copy.deepcopy(human_message)
                    .format(question=demo["question"], fact_before_filter=demo["fact_before_filter"])
                    .text(),
                )
            )
            messages.append(
                ("ai", copy.deepcopy(ai_message).format(fact_after_filter=demo["fact_after_filter"]).text())
            )
        messages.append(copy.deepcopy(human_message))
        all_messages = [system_message]
        all_messages.extend(messages)
        logger.debug(f"DSPyRerankPrompt: {all_messages}")
        prompt_template = ChatPromptTemplate.from_messages(all_messages)
        return prompt_template
