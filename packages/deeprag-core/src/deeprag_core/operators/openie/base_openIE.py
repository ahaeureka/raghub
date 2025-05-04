from abc import abstractmethod

from deeprag_core.schemas.openie_mdoel import OpenIEModel
from deeprag_core.utils.class_meta import SingletonRegisterMeta


class BaseOpenIE(metaclass=SingletonRegisterMeta):
    name = ""

    @abstractmethod
    def extract(self, text: str, lang: str = "zh") -> OpenIEModel:
        """
        Extracts triples from the given text and returns an OpenIEModel object.

        Args:
           text (str): The input text from which triples are to be extracted.

        Returns:
          OpenIEModel: An object containing the extracted triples.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
