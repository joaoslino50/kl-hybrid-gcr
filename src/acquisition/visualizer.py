"""Interactive NetworkX -> PyVis HTML graph visualization for KG subgraphs and reasoning paths."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from pyvis.network import Network

from src.utils.types import SubgraphData, Triple

logger = logging.getLogger(__name__)


def export_subgraph_html(
    subgraph: SubgraphData,
    output_path: str | Path,
    highlight_paths: Sequence[Sequence[Triple]] | None = None,
    height: str = "750px",
    width: str = "100%",
    notebook: bool = False,
) -> Path:
    """Convert a `SubgraphData` object to NetworkX and export an interactive PyVis `.html` graph.

    Node styling:
        - Question entities (`q_entities`): Blue (`#2563eb`)
        - Answer entities (`a_entities`): Green (`#16a34a`)
        - Entities on highlighted reasoning paths: Amber (`#f59e0b`)
        - Other KG entities: Slate (`#64748b`)

    Args:
        subgraph: Input `SubgraphData` instance.
        output_path: Destination `.html` file path.
        highlight_paths: Optional generated reasoning paths (`list[list[Triple]]`) to emphasize.
        height: Canvas CSS height.
        width: Canvas CSS width.
        notebook: Whether rendering inside a Jupyter notebook.

    Returns:
        Path: Resolved path to the written `.html` file.
    """
    target_path = Path(output_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    nx_graph = subgraph.to_networkx()

    highlighted_edges: set[tuple[str, str]] = set()
    highlighted_nodes: set[str] = set()
    if highlight_paths:
        for path in highlight_paths:
            for triple in path:
                highlighted_edges.add((triple.head, triple.tail))
                highlighted_nodes.add(triple.head)
                highlighted_nodes.add(triple.tail)

    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=notebook,
        cdn_resources="remote",
        heading=f"[{subgraph.question_id}] {subgraph.question}",
    )
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=140, spring_strength=0.08)

    for node, attrs in nx_graph.nodes(data=True):
        node_id = str(node)
        is_q = bool(attrs.get("is_q_entity", False))
        is_a = bool(attrs.get("is_a_entity", False))
        on_path = node_id in highlighted_nodes

        if is_q:
            color = "#2563eb"
            role = "Question Entity"
            size = 28
        elif is_a:
            color = "#16a34a"
            role = "Answer Entity"
            size = 28
        elif on_path:
            color = "#f59e0b"
            role = "Reasoning Path Entity"
            size = 22
        else:
            color = "#64748b"
            role = "KG Entity"
            size = 18

        net.add_node(
            node_id,
            label=node_id,
            title=f"{node_id} ({role})",
            color=color,
            size=size,
        )

    for head, tail, attrs in nx_graph.edges(data=True):
        head_id, tail_id = str(head), str(tail)
        relation_label = str(attrs.get("relation", ""))
        is_highlighted = (head_id, tail_id) in highlighted_edges

        net.add_edge(
            head_id,
            tail_id,
            label=relation_label,
            title=f"{head_id} -> {relation_label} -> {tail_id}",
            color="#d97706" if is_highlighted else "#94a3b8",
            width=3.5 if is_highlighted else 1.5,
            arrows="to",
        )

    net.write_html(str(target_path), notebook=notebook)
    logger.info(
        "Exported interactive subgraph HTML (%d nodes, %d edges) to %s",
        nx_graph.number_of_nodes(),
        nx_graph.number_of_edges(),
        target_path,
    )
    return target_path

