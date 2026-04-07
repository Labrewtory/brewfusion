# 🍺 BrewFusion: Graph-Conditioned Diffusion for Beer Recipe Generation

BrewFusion is a state-of-the-art generative AI framework that designs beer recipes by combining **Graph Neural Networks (GNNs)** with **Diffusion Transformers (DiT)**. 

Unlike traditional text-based natural language generators that rely solely on surface-level statistical word frequency, BrewFusion deeply understands the **fundamental chemical, physical, and historical relationships** between brewing ingredients.

---

## ✨ Key Features

- **Graph-Conditioned Latent Space:** Learns ingredient relationships from a massive heterogeneous graph built from 170,000+ real-world recipes. The graph meticulously maps over 21,000 distinct ingredients, hops, and yeasts alongside their complex interactions (NPMI co-occurrence).
- **Chemical Intuition via PubChem:** Encodes true chemical structures (SMILES) of 120+ vital brewing compounds (Terpenes, Esters, Phenols, Maillard products). Through the integration of **CSP (Chemical Structure Prediction) Loss**, the GNN connects >46,000 chemical edges, teaching the model that *Cascade* and *Centennial* hops share *Linalool* and are thus chemically substitutable.
- **Hybrid Token Embedding:** A unified architecture that bridges text and chemistry. Structural sequence markers (e.g., `[MALT]`, `<KG>`) learn standard embeddings, while ingredient words directly inject **frozen continuous GNN vectors** into the DiT's cross-attention map, resolving OOV (Out-of-Vocabulary) issues inherently.
- **Continuous Latent Diffusion (DiT):** Generates robust, timeline-accurate recipes (Mash $\\rightarrow$ Boil $\\rightarrow$ Ferment $\\rightarrow$ Dry Hop) through noise diffusion. Precisely tunable using user-defined scalar metrics (ABV, IBU, Color) controlled by an **AdaLN-Zero** conditioning block.

---

## 🏗 Architecture Overview

1. **Knowledge Extraction (GNN):** 
   Parses raw JSON data and PubChem chemical signatures into a PyTorch Geometric (PyG) Heterogeneous Graph. A `HeteroGNNEncoder` compresses this gigantic topological and chemical network into a rich 64D latent space.
2. **Specialized Tokenization:** 
   Recipes are flattened into time-aware sequence strings. A custom BPE Tokenizer protects critical grammatical markers from fragmentation.
3. **Diffusion Denoising (DiT):** 
   To generate a recipe, the DiT starts with pure Gaussian noise and iteratively denoises it. The `HybridTokenEmbedding` aligns the text tokens with the 64D GNN space, allowing the AdaLN-Zero mechanism to force the recipe to meet strict Target ABV and IBU conditions.

---

## 📁 Repository Structure

```text
brewfusion/
├── data/
│   ├── chemistry/             # Expanded SMILES compounds via PubChem API
│   ├── graph/                 # Trained HeteroData graph & GNN Embeddings
│   └── processed/             # Tokenized sequence caches for Diffusion
├── scripts/
│   ├── data_collection/       # JSON parsers, sequence generators, tokenizers
│   ├── training/              # train_gnn.py, train_dit.py
│   └── inference/             # generate.py (Recipe Generation CLI)
└── src/brewfusion/
    ├── chem/                  # RDKit Morgan Fingerprint processing
    ├── data/                  # DB mapping and Normalizers
    ├── graph/                 # Graph builders and PyG schema definitions
    └── models/
        ├── gnn_encoder.py     # Graph Neural Net & CSP Layer
        ├── dit_brewfusion.py  # Diffusion Transformer (AdaLN-Zero)
        ├── hybrid_embedding.py# GNN-BPE Injection Layer
        └── scheduler.py       # DDPM Noise Scheduler
```

---

## 🚀 Installation & Setup

BrewFusion requires `uv` for lightning-fast dependency management and `PyTorch` with CUDA support.

## 🛠 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Labrewtory/brewfusion.git
cd brewfusion

# 2. Sync dependencies using uv
uv sync

# 3. Verify tests
uv run pytest tests/
```

## 📦 Model Weights (Github Releases)
Since the DiT Checkpoint and GNN Embedding vectors exceed standard Github file size limits, they are packaged into Github Releases.
You can instantaneously download and extract them via our turnkey script:
```bash
uv run python scripts/download_weights.py
```
This will download `brewfusion_weights_v1.tar.gz` and expand it into the `data/` directory natively.

---

## 💻 Usage Pipeline

### 1. Training from scratch
To rebuild the graph memory embeddings and DiT from scratch:
```bash
uv run python scripts/training/train_gnn.py
```

### 2. Train the Diffusion Transformer (DiT)
Train the generative step. The model will automatically inject the frozen GNN embeddings into its architecture.
```bash
uv run python scripts/training/train_dit.py
```

### 3. Generate a Recipe
Use the trained DiT model to sample an entirely new recipe from noise, specifying your desired constraints:
```bash
uv run python scripts/inference/generate.py --abv 6.5 --ibu 45.0 --color 15.0 --cfg 3.0 --num 1
```

---

---

## 📚 Acknowledgements & References

BrewFusion is built upon incredible foundational research and open datasets. If you use this repository, please acknowledge the following works:

1. **FlavorDiffusion (Seo et al., 2025):** The NPMI co-occurrence edge logic inside our Heterogeneous Graph is directly inspired by tracking how naturally ingredients complement each other across varying topologies.
2. **DiT (Peebles & Xie, 2023):** "Scalable Diffusion Models with Transformers" serves as the architectural foundation for the continuous reverse diffusion process (AdaLN-Zero) used to generate recipes. 
3. **PubChem & FlavorDB:** The rich 64D chemical Embeddings generated by the GNN are powered by exact molecular SMILES directly fetched from the National Center for Biotechnology Information (NCBI) PubChem Database and inspired by flavor mapping networks from FlavorDB.
4. **AI-Assisted Engineering:** The entire project architecture—spanning from the theoretical formulation of the Hybrid Token Embedding to the Python implementation of the Graph-Conditioned DiT logic—was collaboratively researched, designed, and strictly engineered in partnership with an advanced Agentic AI Coding Assistant.

*For detailed insights into the graph generation, loss functions, and chemical normalization logic, please explore the source code inside `src/brewfusion/`.*
