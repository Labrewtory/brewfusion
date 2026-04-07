import json
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
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
            
    # -- 4. Save Interactive PyVis HTML --
    net = Network(height="800px", width="100%", bgcolor="#f8f9fa", font_color="#333333", directed=False)
    net.force_atlas_2based(spring_length=150, damping=0.9, overlap=1)
    # Enable Physics
    net.toggle_physics(True)
    
    for idx, node in enumerate(G.nodes()):
        t = G.nodes[node].get("type", "")
        w = G.nodes[node].get("weight", 10)
        c = color_map[idx]
        sw = size_map[idx]
        
        # Scale for pyvis rendering (size mapping)
        if t == "style":
            sz = 45
            border = "#d84315"
            icon = "🍺 "
        elif t == "hop":
            sz = 20 + w // 20
            border = "#2e7d32"
            icon = "🌿 "
        elif t == "malt":
            sz = 20 + w // 20
            border = "#ff8f00"
            icon = "🌾 "
        elif t == "compound":
            sz = 25
            border = "#0277bd"
            icon = "🧪 "
        else:
            sz = 15
            border = "grey"
            icon = ""
            
        net.add_node(
            node, 
            label=f"{icon}{node}", 
            title=f"Type: {t.title()}\\nWeight: {w}", 
            size=sz,
            color={"background": c, "border": border, "highlight": {"background": c, "border": "#212121"}},
            font={"color": "#212121", "size": 16 if t=="style" else 12, "face": "sans-serif"}
        )
        
    for u, v, data in G.edges(data=True):
        ew = data.get("weight", 1)
        net.add_edge(u, v, value=ew, color="#cfd8dc", title=f"Co-occurrence / Mapping: {ew}")
        
    out_html_path = root / f"style_graph_{style_name.replace(' ', '_')}.html"
    net.save_graph(str(out_html_path))
    
    with open(out_html_path, "r", encoding="utf-8") as f:
        html_str = f.read()
        
    return html_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="Stout")
    args = parser.parse_args()
    build_style_subgraph(args.style)
