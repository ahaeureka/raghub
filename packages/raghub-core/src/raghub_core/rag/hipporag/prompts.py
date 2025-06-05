import copy
import json
import os
from typing import Any, Dict, List, Optional

from langchain.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from raghub_core.prompts.base_prompt import BasePrompt
from raghub_core.schemas.dspy_filter_model import DSPyFilterPromptModel


def get_query_instruction(linking_method, lang="en"):
    en_instructions = {
        "ner_to_node": "Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.']}\nQuery: ",  # noqa: E501
        "query_to_node": "Given a question, retrieve relevant phrases that are mentioned in this question.']}\nQuery: ",
        "query_to_fact": "Given a question, retrieve relevant triplet facts that matches this question.']}\nQuery: ",
        "query_to_sentence": "Given a question, retrieve relevant sentences that best answer the question.']}\nQuery: ",
        "query_to_passage": "Given a question, retrieve relevant documents that best answer the question.']}\nQuery: ",
    }
    zh_instructions = {
        "ner_to_node": "给定一个短语，检索与此短语最匹配的同义词或相关短语。\n查询:",
        "query_to_node": "给定一个问题，检索该问题中提到的相关短语。\n查询:",
        "query_to_fact": "给定一个问题，检索与此问题匹配的相关三元组事实。\n查询:",
        "query_to_sentence": "给定一个问题，检索最能回答该问题的相关句子。\n查询:",
        "query_to_passage": "给定一个问题，检索最能回答该问题的相关文档。\n查询:",
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
    raise FileNotFoundError(f"File not found: {best_dspy_prompt_path}")


class DSPyRerankPrompt(BasePrompt):
    def __init__(self, best_dspy_path: str, prompts: Optional[List[DSPyFilterPromptModel]] = None):
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
        prompt_template = ChatPromptTemplate.from_messages(all_messages)
        return prompt_template
