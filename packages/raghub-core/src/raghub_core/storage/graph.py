from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from raghub_core.schemas.document import Document
from raghub_core.utils.class_meta import SingletonRegisterMeta


class GraphStorage(metaclass=SingletonRegisterMeta):
    name = ""

    # _registry: Dict[str, Type["GraphStorage"]] = {}  # 注册表
    @abstractmethod
    def init(self):
        raise NotImplementedError("Subclasses should implement this `init` method.")

    @abstractmethod
    def add_new_edges(self, label: str, node_to_node_stats: Dict[Tuple[str, str], float]) -> None:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get_by_ids(self, ids: List[str]) -> List[Document]:
        raise NotImplementedError("Subclasses should implement this method.")

    # @abstractmethod
    # def vs(self, name: str = "") -> List[str] | Dict[str, List[str]]:
    #     raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def add_vertices(self, label: str, nodes: List[Dict[str, Any]]) -> None:
        raise NotImplementedError("Subclasses should implement this method.")

    # @abstractmethod
    # def edge_seq(self, name: str = "") -> List[str] | Dict[str, List[str]]:
    #     raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def vertices_count(self, label: str) -> int:
        raise NotImplementedError("Subclasses should implement this `vertices_count` method.")

    @abstractmethod
    def personalized_pagerank(
        self,
        label: str,
        vertices_with_weight: Dict[str, float],
        damping: float = 0.85,
        top_k: int = 10,
        **kwargs: Any,
    ) -> Dict[str, float]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def select_vertices(self, label, attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def delete_vertices(self, label: str, keys: List[str]) -> None:
        """Delete vertices from the graph."""
        raise NotImplementedError("Subclasses should implement this method.")

    def transform_logic_operators(self, attr: Dict[str, Any]) -> str:
        """
        Transform logic operators in the attribute dictionary to Cypher syntax.

        Keyword arguments can be used to filter the vertices based on their
        attributes. The name of the keyword specifies the name of the attribute
        and the filtering operator, they should be concatenated by an
        underscore (C{_}) character. Attribute names can also contain
        underscores, but operator names don't, so the operator is always the
        largest trailing substring of the keyword name that does not contain
        an underscore. Possible operators are:

          - C{eq}: equal to

          - C{ne}: not equal to

          - C{lt}: less than

          - C{gt}: greater than

          - C{le}: less than or equal to

          - C{ge}: greater than or equal to

          - C{in}: checks if the value of an attribute is in a given list

          - C{notin}: checks if the value of an attribute is not in a given
            list

        For instance, if you want to filter vertices with a numeric C{age}
        property larger than 200, you have to write:

          >>> select(age_gt=200)  # doctest: +SKIP

        Similarly, to filter vertices whose C{type} is in a list of predefined
        types:

          >>> list_of_types = ["HR", "Finance", "Management"]
          >>> select(type_in=list_of_types)  # doctest: +SKIP

        If the operator is omitted, it defaults to C{eq}. For instance, the
        following selector selects vertices whose C{cluster} property equals
        to 2:

          >>> select(cluster=2)  # doctest: +SKIP
        Args:
            attr: Dictionary of attributes with logic operators
        Returns:
        """
        if not attr:
            return ""
        conditions = []
        for key, value in attr.items():
            if "_" in key:
                attr_name, operator = key.rsplit("_", 1)
                if operator == "eq":
                    conditions.append(f"n.{attr_name} = ${key}")
                elif operator == "ne":
                    conditions.append(f"n.{attr_name} <> ${key}")
                elif operator == "lt":
                    conditions.append(f"n.{attr_name} < ${key}")
                elif operator == "gt":
                    conditions.append(f"n.{attr_name} > ${key}")
                elif operator == "le":
                    conditions.append(f"n.{attr_name} <= ${key}")
                elif operator == "ge":
                    conditions.append(f"n.{attr_name} >= ${key}")
                elif operator == "in":
                    conditions.append(f"n.{attr_name} IN ${key}")
                elif operator == "notin":
                    conditions.append(f"n.{attr_name} NOT IN ${key}")
            else:
                conditions.append(f"n.{key} = ${key}")
        if len(conditions) == 0:
            return ""
        if len(conditions) == 1:
            return conditions[0]
        return " AND ".join(conditions)
