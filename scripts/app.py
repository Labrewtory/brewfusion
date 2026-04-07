import streamlit as st
import time
import sys
import json
from pathlib import Path

# Add project root to sys.path so we can import brewfusion
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from brewfusion.config import PROJECT_ROOT as CFG_ROOT  # noqa: E402
from scripts.inference.generate import load_model, generate  # noqa: E402


# --- Model Loading ---
@st.cache_resource(show_spinner="Loading BrewFusion DiT & GNN Embeddings...")
def get_model_components():
    checkpoint_path = CFG_ROOT / "data" / "models" / "dit_best.pt"
    tokenizer_path = CFG_ROOT / "src" / "brewfusion" / "data" / "brew_tokenizer.json"

    if not checkpoint_path.exists() or not tokenizer_path.exists():
        st.error("Model weights not found! Please run training first.")
        st.stop()

    return load_model(checkpoint_path, tokenizer_path)


def format_recipe_readable(raw_text: str) -> str:
    """Format raw BPE token sequences into a human-readable structure."""
    text = raw_text.replace(" _", "_").replace("_ ", "_").replace("  ", " ")

    # Structural Markers
    text = text.replace("[BOIL_START]", "\n\n🔥 **Boil Start:**")
    text = text.replace("[BOIL]", "\n\n⏱️ **Boil Process:**")
    text = text.replace("[HOP]", "\n\n🪴 **Hop Addition:**")
    text = text.replace("[MASH_STEP]", "\n\n💧 **Mash Step:**")
    text = text.replace("[FERMENTATION]", "\n\n🧪 **Fermentation:**")
    text = text.replace("[TARGET_IBU]", "\n\n🎯 **Stats/Targets:** TARGET_IBU")

    # Units
    text = text.replace("<KG>", "kg")
    text = text.replace("<G>", "g")
    text = text.replace("<MIN>", "min")
    text = text.replace("<L>", "L")

    # Clean up material tokens (e.g. AMERICAN___S -> American S)
    words = text.split(" ")
    cleaned_words = []
    for w in words:
        if "_" in w and not w.startswith("[") and not w.endswith("]"):
            cw = " ".join(part.capitalize() for part in w.split("_") if part)
            cleaned_words.append(cw)
        else:
            cleaned_words.append(w)

    return " ".join(cleaned_words)


# Load real style registry (V1.1)
try:
    registry_path = CFG_ROOT / "src" / "brewfusion" / "data" / "style_registry.json"
    with open(registry_path, "r", encoding="utf-8") as f:
        _style_dict = json.load(f)
        STYLE_REGISTRY = {idx: f"{idx} - {name}" for name, idx in _style_dict.items()}
        STYLE_REGISTRY = dict(sorted(STYLE_REGISTRY.items()))
except Exception:
    STYLE_REGISTRY = {0: "0 - Default"}

# --- Page Config ---
st.set_page_config(
    page_title="BrewFusion | Sequence-Based Recipe Generation",
    page_icon="🍺",
    layout="wide",
)

st.title("🍺 BrewFusion: AI Brewing Engineer")

# ─────────────────────────────────────────────
# Mode Selector (top of sidebar, always visible)
# ─────────────────────────────────────────────
mode = st.sidebar.radio(
    "Navigation",
    ["🚀 Recipe Generator", "🕸️ GNN Style Graph"],
    label_visibility="collapsed",
)

# ═════════════════════════════════════════════
# MODE 1 — Recipe Generator
# ═════════════════════════════════════════════
if mode == "🚀 Recipe Generator":
    st.markdown("""
    Adjust the target thermodynamic and flavor constraints in the sidebar, 
    and the **Graph-Conditioned Diffusion Transformer (DiT)** will generate 
    a complete timeline of ingredients using hybrid chemical embeddings.
    """)

    # --- Sidebar controls (only for this mode) ---
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Target Constraints")
    st.sidebar.markdown("Configure the scalar physics metrics:")

    abv = st.sidebar.slider(
        "Alcohol By Volume (ABV %)",
        min_value=0.0, max_value=15.0, value=6.0, step=0.1,
    )
    ibu = st.sidebar.slider(
        "International Bitterness Units (IBU)",
        min_value=0.0, max_value=120.0, value=35.0, step=1.0,
    )
    color = st.sidebar.slider(
        "Color (SRM)",
        min_value=0.0, max_value=40.0, value=4.0, step=0.1,
    )

    st.sidebar.header("🏷️ Categorical Style")
    style_display = st.sidebar.selectbox(
        "Style Category (Loaded from Database)",
        options=list(STYLE_REGISTRY.keys()),
        format_func=lambda x: STYLE_REGISTRY[x],
        index=14 if 14 in STYLE_REGISTRY else 0,
    )
    style_idx = st.sidebar.number_input(
        "Or manually set Style Index",
        min_value=0,
        max_value=max(list(STYLE_REGISTRY.keys())) if STYLE_REGISTRY else 179,
        value=style_display,
        help="Overrides dropdown if changed manually.",
    )

    st.sidebar.header("🧠 Generation Logic")
    cfg_scale = st.sidebar.slider(
        "CFG Guidance Scale",
        min_value=1.0, max_value=10.0, value=3.5, step=0.5,
        help="Higher values force the model to aggressively obey scalar constraints.",
    )
    num_samples = st.sidebar.number_input(
        "Number of recipes", min_value=1, max_value=5, value=1,
    )

    # --- Main area: Generation ---
    loaded_components = get_model_components()

    if st.button("🍻 Generate Recipe Matrix", type="primary"):
        with st.spinner(f"Diffusing {num_samples} recipes out of chemical noise..."):
            start_time = time.time()
            try:
                results = generate(
                    abv=abv,
                    ibu=ibu,
                    color=color,
                    style_idx=style_idx,
                    cfg_scale=cfg_scale,
                    num_samples=num_samples,
                    loaded_components=loaded_components,
                )

                elapsed = time.time() - start_time
                st.success(f"Generation completed in {elapsed:.2f} seconds!")

                for i, recipe in enumerate(results):
                    st.subheader(f"Recipe Option #{i + 1}")
                    tab1, tab2 = st.tabs(["📝 Readable Format", "⚙️ Raw Tokens"])
                    with tab1:
                        st.markdown(format_recipe_readable(recipe))
                    with tab2:
                        st.code(recipe, language="text")

                    if color <= 6.0 and "DARK_" in recipe:
                        st.warning(
                            "⚠️ Dark malts despite low color constraint. "
                            "Consider increasing CFG scale."
                        )
                    elif color <= 6.0 and style_idx == 100 and "DARK_" not in recipe:
                        st.info(
                            "💡 **White Stout Anomaly:** The model stripped dark "
                            "malts to obey the pale constraint!"
                        )

            except Exception as exc:
                st.error(f"Error during generation: {exc}")

# ═════════════════════════════════════════════
# MODE 2 — GNN Style Physical Graph
# ═════════════════════════════════════════════
else:
    st.header("🔬 Heterogeneous Flavor Graph")
    st.markdown("""
    Explore the interconnected physics beneath each beer style.  
    The **Graph Neural Network (GNN)** learns 64-dimensional embeddings from 
    this exact network structure — styles, hops, malts, and flavor compounds 
    like Myrcene — physically grounding the DiT generation engine.
    """)

    st.info(
        "💡 **Tip:** Drag nodes to play with the Barnes-Hut gravity simulation, "
        "or zoom/scroll to explore deep connections."
    )

    # --- Sidebar controls (only for this mode) ---
    st.sidebar.markdown("---")
    st.sidebar.header("🔬 Graph Controls")

    graph_style_idx = st.sidebar.selectbox(
        "Select a Style to Map",
        options=list(STYLE_REGISTRY.keys()),
        format_func=lambda x: STYLE_REGISTRY[x],
        index=14 if 14 in STYLE_REGISTRY else 0,
        key="graph_style_selector",
    )
    max_nodes = st.sidebar.slider(
        "Max Nodes (per type)",
        min_value=10, max_value=60, value=35, step=5,
        help="Limit the number of ingredient/compound nodes displayed.",
    )

    st.info(
        "💡 **Tip:** Drag nodes to play with the Barnes-Hut gravity simulation, "
        "or zoom/scroll to explore deep connections."
    )

    if st.button("🌌 Render Graph", type="primary"):
        with st.spinner("Extracting neighbourhood from PyG HeteroData..."):
            from scripts.visualize_style_graph import build_style_subgraph
            import streamlit.components.v1 as components

            style_name = STYLE_REGISTRY[graph_style_idx].split(" - ", 1)[1]
            try:
                html_str = build_style_subgraph(style_name, max_nodes=max_nodes)
                st.success(f"Rendered subgraph for **{style_name}**!")
                components.html(html_str, height=850)
            except Exception as exc:
                st.error(f"Could not render graph: {exc}")

