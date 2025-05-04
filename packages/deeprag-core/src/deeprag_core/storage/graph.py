from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from deeprag_core.schemas.document import Document
from deeprag_core.utils.class_meta import SingletonRegisterMeta


class GraphStorage(metaclass=SingletonRegisterMeta):
    name = ""

    # _registry: Dict[str, Type["GraphStorage"]] = {}  # 注册表
    @abstractmethod
    def init(self):
        raise NotImplementedError("Subclasses should implement this `init` method.")

    @abstractmethod
    def add_edges(self, edges: List[Tuple[str, str]], weights: List[float]) -> None:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get_by_ids(self, ids: List[str]) -> List[Document]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def vs(self, name: str = "") -> List[str] | Dict[str, List[str]]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def add_vertices(self, n: int, attributes: Dict[str, List[Any]]) -> None:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def edge_seq(self, name: str = "") -> List[str] | Dict[str, List[str]]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def vertices_count(self) -> int:
        raise NotImplementedError("Subclasses should implement this `vertices_count` method.")

    @abstractmethod
    def personalized_pagerank(
        self,
        vertices: Any | None = None,
        directed: bool = True,
        damping: float = 0.85,
        weights: Any | None = None,
        reset: Any | None = None,
        arpack_options: Any | None = None,
        implementation: str = "prpack",
        top_k: int = 10,
    ) -> Dict[str, float]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def select_vertices(self, **attrs) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses should implement this method.")
