from raghub_core.schemas.graph_model import GraphCommunity, GraphEdge, GraphModel, GraphVertex, Namespace
from raghub_core.utils.misc import compute_mdhash_id


class GraphHelper:
    @staticmethod
    def format_vertex(vertex: GraphVertex, concise: bool = False) -> str:
        """Format a vertex to string."""
        line_break = "\n"

        if concise:
            return f"({vertex.name})"
        else:
            desc = vertex.description or {}
            return f"({vertex.content}:{line_break.join(list(desc.values()))})"  # noqa: E501

    @staticmethod
    def format_edge(edge: GraphEdge, concise: bool = False) -> str:
        """Format an edge to string."""
        fmt = ""
        line_break = "\n"
        if concise:
            fmt = f"{edge.source} -[{edge.relation_type}]-> {edge.target}"
        else:
            desc = edge.description or {}
            fmt = f"({edge.source_content}) -[{edge.relation_type}: {line_break.join(list(desc.values()))}]-> ({edge.target_content})"  # noqa: E501
        return fmt

    @staticmethod
    def format_graph(graph: GraphModel, entities_only=False) -> str:
        """Format graph to string."""
        vs_str = "\n".join(GraphHelper.format_vertex(v) for v in graph.vertices)
        es_str = "\n".join(GraphHelper.format_edge(e) for e in graph.edges if e.source and e.target)
        if entities_only:
            return f"Entities:\n{vs_str}" if vs_str else ""
        else:
            return f"Entities:\n{vs_str}\n\nRelationships:\n{es_str}" if (vs_str or es_str) else ""

    @staticmethod
    def format_community(community: GraphCommunity) -> str:
        entities = []
        relationships = []
        line_break = "\n"
        for ent in community.graph.vertices:
            desc = ent.description or {}
            entities.append(f"({ent.content}:{line_break.join(desc)})")
        for ref in community.graph.edges:
            desc = ref.description or {}
            relationships.append(
                f"({ref.source})-[{ref.relation_type}:{line_break.join(list(desc.values()))}]->({ref.target})"
            )
        entities_str = "\n".join(entities)
        relationships_str = "\n".join(relationships)
        return f"Entities:\n{entities_str}\n\nRelationships:\n{relationships_str}\n\n"

    @staticmethod
    def generate_vertex_id(index_name: str, name: str) -> str:
        """Generate a unique ID for an entity based on its name and description."""
        return compute_mdhash_id(index_name, name, Namespace.ENTITY.value)

    @staticmethod
    def generate_edge_id(index_name: str, source: str, relation: str, target: str) -> str:
        """Generate a unique ID for an edge based on source, target and relation type."""
        return compute_mdhash_id(index_name, f"{source}#{relation}#{target}", Namespace.FACT.value)
