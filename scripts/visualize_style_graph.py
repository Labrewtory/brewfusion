"""Visualize a style's GNN neighbourhood from the real PyG HeteroData graph.

Loads ``brewfusion_hetero_graph.pt``, extracts the 1-hop subgraph around
the requested beer_style node, and renders it as an interactive PyVis HTML.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import networkx as nx
import torch
from pyvis.network import Network


def _load_heterodata() -> object:
    """Load the saved HeteroData graph (build it first if missing)."""
    root = Path(__file__).resolve().parent.parent
    graph_path = root / "data" / "graph" / "brewfusion_hetero_graph.pt"

    if not graph_path.exists():
        # Auto-build if the file is absent
        from brewfusion.graph.builder import build_graph, save_graph  # noqa: F811

        data = build_graph()
        save_graph(data, graph_path)
    else:
        data = torch.load(graph_path, weights_only=False)

    return data


# ── Colour / icon palette per node type ──────────────────────────
_PALETTE = {
    "beer_style": {"bg": "#ff5722", "border": "#d84315", "icon": "🍺", "base_sz": 50},
    "ingredient": {"bg": "#ffc107", "border": "#ff8f00", "icon": "🌾", "base_sz": 18},
    "hop":        {"bg": "#4caf50", "border": "#2e7d32", "icon": "🌿", "base_sz": 18},
    "yeast":      {"bg": "#ab47bc", "border": "#7b1fa2", "icon": "🧫", "base_sz": 22},
    "compound":   {"bg": "#03a9f4", "border": "#0277bd", "icon": "🧪", "base_sz": 20},
}


def build_style_subgraph(style_name: str, max_nodes: int = 35, n_hops: int = 2) -> str:
    """Extract the n-hop neighbourhood of *style_name* and return PyVis HTML.

    Args:
        style_name: Beer style to centre the graph on.
        max_nodes: Max nodes per type in the 1-hop ring.
        n_hops: Depth of traversal (1, 2, or 3).
    """

    data = _load_heterodata()

    # ── Resolve style index ──────────────────────────────────────
    style_names: list[str] = data["beer_style"].names
    style_idx: int | None = None
    for i, name in enumerate(style_names):
        if style_name.lower() in name.lower():
            style_idx = i
            break

    if style_idx is None:
        raise ValueError(f"Style '{style_name}' not found in graph ({len(style_names)} styles)")

    resolved_name = style_names[style_idx]

    # ── HOP 1: style → ingredient / hop / yeast ──────────────────
    neighbour_lists: dict[str, list[tuple[int, str]]] = defaultdict(list)

    edge_meta = [
        ("uses_grain", "ingredient"),
        ("uses_hop", "hop"),
        ("uses_yeast", "yeast"),
        ("uses_adjunct", "ingredient"),
    ]

    for rel, dst_type in edge_meta:
        et = ("beer_style", rel, dst_type)
        if et not in data.edge_types:
            continue
        ei = data[et].edge_index
        mask = ei[0] == style_idx
        dst_indices = ei[1, mask].tolist()
        dst_names = data[dst_type].names
        for di in dst_indices:
            neighbour_lists[dst_type].append((di, dst_names[di]))

    # Deduplicate & keep top-N per type
    budget = {
        "ingredient": int(max_nodes * 0.35),
        "hop":        int(max_nodes * 0.30),
        "yeast":      int(max_nodes * 0.15),
        "compound":   int(max_nodes * 0.20),
    }

    top_neighbours: dict[str, list[tuple[int, str, int]]] = {}
    for ntype, items in neighbour_lists.items():
        counts: dict[str, int] = defaultdict(int)
        idx_map: dict[str, int] = {}
        for idx, name in items:
            counts[name] += 1
            idx_map[name] = idx
        ranked = sorted(counts.items(), key=lambda x: -x[1])[: budget.get(ntype, 5)]
        top_neighbours[ntype] = [(idx_map[n], n, c) for n, c in ranked]

    # ── HOP 2: ingredient/hop/yeast → compound ───────────────────
    compound_src: dict[str, set[str]] = defaultdict(set)

    if n_hops >= 2:
        compound_edge_meta = [
            ("ingredient", "contains", "compound"),
            ("hop", "contains", "compound"),
            ("yeast", "produces", "compound"),
        ]

        compound_counts: dict[str, int] = defaultdict(int)
        compound_idx_map: dict[str, int] = {}

        for src_type, rel, dst_type in compound_edge_meta:
            et = (src_type, rel, dst_type)
            if et not in data.edge_types:
                continue
            ei = data[et].edge_index
            src_names_list = data[src_type].names
            dst_names_list = data[dst_type].names

            selected_src_indices = {idx for idx, _, _ in top_neighbours.get(src_type, [])}

            for col in range(ei.shape[1]):
                si, di = ei[0, col].item(), ei[1, col].item()
                if si in selected_src_indices:
                    cname = dst_names_list[di]
                    compound_counts[cname] += 1
                    compound_idx_map[cname] = di
                    compound_src[cname].add(src_names_list[si])

        ranked_compounds = sorted(compound_counts.items(), key=lambda x: -x[1])[: budget["compound"]]
        top_neighbours["compound"] = [(compound_idx_map[n], n, c) for n, c in ranked_compounds]

    # ── HOP 3: compound ↔ similar compound & ingredient co-occurrence
    hop3_edges: list[tuple[str, str, int]] = []

    if n_hops >= 3:
        # 3a. compound → similar_to → compound
        et_sim = ("compound", "similar_to", "compound")
        if et_sim in data.edge_types:
            ei = data[et_sim].edge_index
            cmp_names = data["compound"].names
            selected_compound_indices = {idx for idx, _, _ in top_neighbours.get("compound", [])}
            new_compounds: dict[str, int] = {}

            for col in range(ei.shape[1]):
                si, di = ei[0, col].item(), ei[1, col].item()
                if si in selected_compound_indices and di not in selected_compound_indices:
                    cname = cmp_names[di]
                    if cname not in new_compounds:
                        new_compounds[cname] = di
                    src_cname = cmp_names[si]
                    hop3_edges.append((src_cname, cname, 1))

            # Add up to 8 new similar compounds
            for cname, cidx in list(new_compounds.items())[:8]:
                top_neighbours.setdefault("compound", []).append((cidx, cname, 1))

        # 3b. ingredient ↔ cooccurs_with → ingredient
        et_cooc = ("ingredient", "cooccurs_with", "ingredient")
        if et_cooc in data.edge_types:
            ei = data[et_cooc].edge_index
            ing_names = data["ingredient"].names
            selected_ing_indices = {idx for idx, _, _ in top_neighbours.get("ingredient", [])}
            cooc_counts: dict[str, int] = defaultdict(int)
            cooc_idx_map: dict[str, int] = {}

            for col in range(ei.shape[1]):
                si, di = ei[0, col].item(), ei[1, col].item()
                if si in selected_ing_indices and di not in selected_ing_indices:
                    iname = ing_names[di]
                    cooc_counts[iname] += 1
                    cooc_idx_map[iname] = di

            # Add top 6 co-occurring ingredients
            cooc_ranked = sorted(cooc_counts.items(), key=lambda x: -x[1])[:6]
            for iname, cnt in cooc_ranked:
                top_neighbours.setdefault("ingredient", []).append((cooc_idx_map[iname], iname, cnt))
                # Find which selected ingredient co-occurs with this new one
                for col in range(ei.shape[1]):
                    si, di = ei[0, col].item(), ei[1, col].item()
                    if di == cooc_idx_map[iname] and si in selected_ing_indices:
                        hop3_edges.append((ing_names[si], iname, 1))
                        break

        # 3c. hop ↔ cooccurs_with → hop
        et_hcooc = ("hop", "cooccurs_with", "hop")
        if et_hcooc in data.edge_types:
            ei = data[et_hcooc].edge_index
            hop_names = data["hop"].names
            selected_hop_indices = {idx for idx, _, _ in top_neighbours.get("hop", [])}
            hcooc_counts: dict[str, int] = defaultdict(int)
            hcooc_idx_map: dict[str, int] = {}

            for col in range(ei.shape[1]):
                si, di = ei[0, col].item(), ei[1, col].item()
                if si in selected_hop_indices and di not in selected_hop_indices:
                    hname = hop_names[di]
                    hcooc_counts[hname] += 1
                    hcooc_idx_map[hname] = di

            hcooc_ranked = sorted(hcooc_counts.items(), key=lambda x: -x[1])[:4]
            for hname, cnt in hcooc_ranked:
                top_neighbours.setdefault("hop", []).append((hcooc_idx_map[hname], hname, cnt))
                for col in range(ei.shape[1]):
                    si, di = ei[0, col].item(), ei[1, col].item()
                    if di == hcooc_idx_map[hname] and si in selected_hop_indices:
                        hop3_edges.append((hop_names[si], hname, 1))
                        break

    # ── Build NetworkX graph ─────────────────────────────────────
    G = nx.Graph()

    # Centre node
    G.add_node(resolved_name, type="beer_style", weight=0)

    for ntype, items in top_neighbours.items():
        for idx, name, count in items:
            G.add_node(name, type=ntype, weight=count)
            if ntype != "compound":
                # Only connect directly to style if it's a 1-hop neighbour
                G.add_edge(resolved_name, name, weight=max(1, count // 5))

    # Compound edges (connect to their source ingredient/hop/yeast)
    for _, cname, _ in top_neighbours.get("compound", []):
        for src_name in compound_src.get(cname, set()):
            if src_name in G:
                G.add_edge(src_name, cname, weight=2)

    # 3-hop edges
    for src, dst, w in hop3_edges:
        if src in G and dst in G:
            G.add_edge(src, dst, weight=w)

    # ── Render PyVis ─────────────────────────────────────────────
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#f8f9fa",
        font_color="#333333",
        directed=False,
    )
    net.force_atlas_2based(spring_length=160, damping=0.9, overlap=1)
    net.toggle_physics(True)

    for node in G.nodes():
        ntype = G.nodes[node].get("type", "")
        w = G.nodes[node].get("weight", 1)
        pal = _PALETTE.get(ntype, _PALETTE["compound"])

        sz = pal["base_sz"] + w // 10
        label = f"{pal['icon']} {node}"
        title = f"Type: {ntype.replace('_', ' ').title()}\\nEdge count: {w}"

        net.add_node(
            node,
            label=label,
            title=title,
            size=sz,
            color={
                "background": pal["bg"],
                "border": pal["border"],
                "highlight": {"background": pal["bg"], "border": "#212121"},
            },
            font={
                "color": "#212121",
                "size": 16 if ntype == "beer_style" else 11,
                "face": "sans-serif",
            },
        )

    for u, v, edata in G.edges(data=True):
        ew = edata.get("weight", 1)
        net.add_edge(u, v, value=ew, color="#cfd8dc", title=f"Weight: {ew}")

    # ── Save and return HTML ─────────────────────────────────────
    root = Path(__file__).resolve().parent.parent
    safe_name = style_name.replace("/", "-").replace("\\", "-").replace(" ", "_")
    out_path = root / f"style_graph_{safe_name}.html"
    net.save_graph(str(out_path))

    with open(out_path, "r", encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────
# View 2: Style Embedding Map (t-SNE of 64D GNN vectors)
# ─────────────────────────────────────────────────────────────────
def build_style_embedding_map() -> str:
    """Project all 180 GNN style embeddings to 2D via t-SNE and render as PyVis."""
    from sklearn.manifold import TSNE
    import numpy as np

    root = Path(__file__).resolve().parent.parent
    emb_path = root / "data" / "graph" / "gnn_embeddings.pt"
    emb_dict = torch.load(emb_path, weights_only=False)
    style_emb = emb_dict["beer_style"]  # [180, 64]

    data = _load_heterodata()
    style_names: list[str] = data["beer_style"].names

    # Normalise to prevent NaN in t-SNE (zero-variance columns crash PCA init)
    emb_np = style_emb.numpy().astype(np.float64)
    std = emb_np.std(axis=0)
    std[std == 0] = 1.0
    emb_np = (emb_np - emb_np.mean(axis=0)) / std

    # t-SNE → 2D
    coords = TSNE(
        n_components=2, perplexity=min(15, len(style_names) - 1),
        random_state=42, init="random", learning_rate="auto",
    ).fit_transform(emb_np)

    # Macro-category colour mapping
    _MACRO = {
        "ipa":     "#4caf50",
        "pale":    "#8bc34a",
        "ale":     "#ff9800",
        "stout":   "#5d4037",
        "porter":  "#795548",
        "lager":   "#2196f3",
        "pilsner": "#03a9f4",
        "wheat":   "#ffc107",
        "weizen":  "#ffc107",
        "belgian": "#e91e63",
        "saison":  "#f06292",
        "sour":    "#ff5252",
        "amber":   "#ff7043",
        "brown":   "#a1887f",
        "barley":  "#6d4c41",
        "scotch":  "#8d6e63",
        "bock":    "#7e57c2",
        "bitter":  "#cddc39",
    }

    def _macro_color(name: str) -> str:
        low = name.lower()
        for key, col in _MACRO.items():
            if key in low:
                return col
        return "#9e9e9e"

    # Scale coords to pyvis-friendly pixel range
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    def _scale(v, vmin, vmax, out_lo=-800, out_hi=800):
        if vmax == vmin:
            return 0.0
        return out_lo + (v - vmin) / (vmax - vmin) * (out_hi - out_lo)

    net = Network(
        height="850px", width="100%", bgcolor="#1e1e2e", font_color="#cdd6f4", directed=False,
    )
    net.toggle_physics(False)  # fixed positions

    for i, name in enumerate(style_names):
        col = _macro_color(name)
        px = float(_scale(coords[i, 0], x_min, x_max))
        py = float(_scale(coords[i, 1], y_min, y_max))
        net.add_node(
            name,
            label=name,
            title=f"{name}\nGNN idx: {i}",
            x=px, y=py,
            size=18,
            color={"background": col, "border": col, "highlight": {"background": "#ffffff", "border": col}},
            font={"color": "#cdd6f4", "size": 10, "face": "sans-serif"},
            fixed=True,
        )

    # Connect styles that share many ingredients (cosine similarity > threshold)
    import torch.nn.functional as F

    sim = F.cosine_similarity(style_emb.unsqueeze(1), style_emb.unsqueeze(0), dim=2)
    threshold = 0.85
    for i in range(len(style_names)):
        for j in range(i + 1, len(style_names)):
            if sim[i, j].item() > threshold:
                net.add_edge(
                    style_names[i], style_names[j],
                    value=1, color={"color": "#45475a", "opacity": 0.3},
                    title=f"Cosine sim: {sim[i, j].item():.3f}",
                )

    out_path = root / "style_embedding_map.html"
    net.save_graph(str(out_path))
    with open(out_path, "r", encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────
# View 3: Filtered Full Graph Overview
# ─────────────────────────────────────────────────────────────────
def build_full_graph_overview(
    top_ingredients: int = 40,
    top_hops: int = 25,
    top_yeasts: int = 15,
) -> str:
    """Render all 180 styles + top-N ingredients/hops/yeasts + all compounds."""

    data = _load_heterodata()

    G = nx.Graph()

    # ── Add ALL style nodes ──────────────────────────────────────
    style_names = data["beer_style"].names
    for name in style_names:
        G.add_node(name, type="beer_style", weight=0)

    # ── Add top-N ingredient/hop/yeast by degree ─────────────────
    def _top_by_degree(ntype: str, edge_types: list, top_n: int) -> set[int]:
        """Return indices of the top-N most-connected nodes of *ntype*."""
        degree = defaultdict(int)
        for et in edge_types:
            if et not in data.edge_types:
                continue
            ei = data[et].edge_index
            # dst side
            _, rel, dst = et
            if dst == ntype:
                for idx in ei[1].tolist():
                    degree[idx] += 1
            src, _, _ = et
            if src == ntype:
                for idx in ei[0].tolist():
                    degree[idx] += 1
        ranked = sorted(degree.items(), key=lambda x: -x[1])[:top_n]
        return {idx for idx, _ in ranked}

    keep_ingredient = _top_by_degree(
        "ingredient",
        [("beer_style", "uses_grain", "ingredient"), ("beer_style", "uses_adjunct", "ingredient")],
        top_ingredients,
    )
    keep_hop = _top_by_degree(
        "hop", [("beer_style", "uses_hop", "hop")], top_hops,
    )
    keep_yeast = _top_by_degree(
        "yeast", [("beer_style", "uses_yeast", "yeast")], top_yeasts,
    )

    for idx in keep_ingredient:
        G.add_node(data["ingredient"].names[idx], type="ingredient", weight=0)
    for idx in keep_hop:
        G.add_node(data["hop"].names[idx], type="hop", weight=0)
    for idx in keep_yeast:
        G.add_node(data["yeast"].names[idx], type="yeast", weight=0)

    # ── Add edges (only between visible nodes) ───────────────────
    edge_configs = [
        (("beer_style", "uses_grain", "ingredient"), style_names, data["ingredient"].names, None, keep_ingredient),
        (("beer_style", "uses_hop", "hop"), style_names, data["hop"].names, None, keep_hop),
        (("beer_style", "uses_yeast", "yeast"), style_names, data["yeast"].names, None, keep_yeast),
        (("beer_style", "uses_adjunct", "ingredient"), style_names, data["ingredient"].names, None, keep_ingredient),
    ]

    for et, src_names, dst_names, src_keep, dst_keep in edge_configs:
        if et not in data.edge_types:
            continue
        ei = data[et].edge_index
        for col in range(ei.shape[1]):
            si, di = ei[0, col].item(), ei[1, col].item()
            if (src_keep is None or si in src_keep) and (dst_keep is None or di in dst_keep):
                sn, dn = src_names[si], dst_names[di]
                if sn in G and dn in G:
                    G.add_edge(sn, dn)

    # ── Render ───────────────────────────────────────────────────
    net = Network(
        height="850px", width="100%", bgcolor="#1e1e2e", font_color="#cdd6f4", directed=False,
    )
    net.force_atlas_2based(spring_length=120, damping=0.9, overlap=0.8)
    net.toggle_physics(True)

    for node in G.nodes():
        ntype = G.nodes[node].get("type", "")
        pal = _PALETTE.get(ntype, _PALETTE["compound"])
        deg = G.degree(node)
        sz = pal["base_sz"] + min(deg, 30)
        net.add_node(
            node,
            label=f"{pal['icon']} {node}",
            title=f"{ntype.replace('_',' ').title()}\nConnections: {deg}",
            size=sz,
            color={"background": pal["bg"], "border": pal["border"]},
            font={"color": "#cdd6f4", "size": 10 if ntype != "beer_style" else 12, "face": "sans-serif"},
        )

    for u, v in G.edges():
        net.add_edge(u, v, color={"color": "#45475a", "opacity": 0.15})

    root = Path(__file__).resolve().parent.parent
    out_path = root / "style_full_graph.html"
    net.save_graph(str(out_path))
    with open(out_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="Stout")
    parser.add_argument("--mode", choices=["subgraph", "embedding", "full"], default="subgraph")
    args = parser.parse_args()

    if args.mode == "subgraph":
        html = build_style_subgraph(args.style)
    elif args.mode == "embedding":
        html = build_style_embedding_map()
    else:
        html = build_full_graph_overview()
    print(f"Generated {len(html)} bytes of HTML")
