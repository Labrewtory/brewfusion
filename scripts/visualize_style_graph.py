import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import argparse

def build_style_subgraph(style_name: str, max_nodes: int = 25):
    root = Path(__file__).resolve().parent.parent
    recipes_txt = root / "recipes_full.txt"
    
    with open(recipes_txt, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    G = nx.Graph()
    G.add_node(style_name, type="style")
    
    hop_counts = defaultdict(int)
    malt_counts = defaultdict(int)
    
    # Simple Hop to Compound Mock mapping for illustration based on known dataset
    # In V2 this would query the exact PyG HeteroData Graph.
    hop_to_compound = {
        "Cascade": ["Myrcene", "Farnesene"],
        "Citra": ["Myrcene", "Linalool", "Geraniol"],
        "Fuggle": ["Farnesene", "Caryophyllene"],
        "Centennial": ["Myrcene", "Linalool"],
        "Magnum": ["Humulene", "Pinene"],
        "Saaz": ["Farnesene"],
        "East Kent Goldings": ["Humulene"],
        "Amarillo": ["Myrcene"],
        "Simcoe": ["Pinene", "Myrcene"]
    }
    
    # 1. Parse recipes matching style
    print(f"Parsing recipes for {style_name}...")
    for recipe in data.values():
        r_style = recipe.get("style", "")
        if isinstance(r_style, str) and style_name.lower() in r_style.lower():
            for hop in recipe.get("hops", []):
                if isinstance(hop, list) and len(hop) > 1:
                    hname = str(hop[1]).strip()
                    if hname: hop_counts[hname] += 1
            for malt in recipe.get("fermentables", []):
                if isinstance(malt, list) and len(malt) > 1:
                    mname = str(malt[1]).strip()
                    if mname: malt_counts[mname] += 1
                    
    # 2. Keep top N
    top_hops = sorted(hop_counts.items(), key=lambda x: -x[1])[:int(max_nodes*0.4)]
    top_malts = sorted(malt_counts.items(), key=lambda x: -x[1])[:int(max_nodes*0.4)]
    
    # 3. Add Edges
    for hop, count in top_hops:
        G.add_node(hop, type="hop", weight=count)
        G.add_edge(style_name, hop, weight=count//10)
        
        # Link compounds if known
        for known_hop, compounds in hop_to_compound.items():
            if known_hop.lower() in hop.lower():
                for comp in compounds:
                    G.add_node(comp, type="compound")
                    G.add_edge(hop, comp, weight=5)
                    
    for malt, count in top_malts:
        G.add_node(malt, type="malt", weight=count)
        G.add_edge(style_name, malt, weight=count//10)
        
    # 4. Plot
    plt.figure(figsize=(14, 10))
    
    # Colors
    color_map = []
    size_map = []
    for node in G.nodes():
        t = G.nodes[node].get("type", "")
        w = G.nodes[node].get("weight", 10)
        
        if t == "style":
            color_map.append("#ff5722") # Orange
            size_map.append(1500)
        elif t == "hop":
            color_map.append("#4caf50") # Green
            size_map.append(400 + w)
        elif t == "malt":
            color_map.append("#ffc107") # Gold
            size_map.append(400 + w)
        elif t == "compound":
            color_map.append("#03a9f4") # Blue
            size_map.append(600)
        else:
            color_map.append("grey")
            size_map.append(500)
            
    pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=size_map, edgecolors="white", linewidths=2)
    nx.draw_networkx_edges(G, pos, edge_color="#b0bec5", alpha=0.6, width=1.5)
    
    # Labels
    font_opts = {"font_size": 11, "font_weight": "bold", "font_family": "sans-serif"}
    nx.draw_networkx_labels(G, pos, **font_opts)
    
    plt.title(f"Heterogeneous Subgraph: [{style_name}]", fontsize=18, fontweight="bold")
    plt.axis('off')
    
    plt.tight_layout()
    out_path = root / f"style_graph_{style_name.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor="#f8f9fa")
    print(f"Graph saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="Stout")
    args = parser.parse_args()
    build_style_subgraph(args.style)
