"""Rebuild the BPE tokenizer with structural tokens as special tokens.

Best practice: All bracket/angle-bracket tokens like [MALT], [HOP], <KG>
must be registered as special tokens so BPE never splits them.
"""

from pathlib import Path

from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def train_brew_tokenizer() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data" / "processed" / "all_sequences.txt"
    out_dir = project_root / "src" / "brewfusion" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "brew_tokenizer.json"

    # ──────────────────────────────────────────────
    # Special tokens: registered FIRST so BPE never splits them.
    # Order matters: IDs are assigned in this order (0, 1, 2, ...).
    # ──────────────────────────────────────────────
    special_tokens = [
        # Control tokens
        "[UNK]",  # 0
        "[PAD]",  # 1
        "[BOS]",  # 2
        "[EOS]",  # 3
        # Recipe structure tokens
        "[START]",  # 4
        "[END]",  # 5
        "[TARGET_ABV]",  # 6
        "[TARGET_IBU]",  # 7
        "[TARGET_COLOR]",  # 8
        "[MASH_STEP]",  # 9
        "[MALT]",  # 10
        "[BOIL_START]",  # 11
        "[BOIL]",  # 12
        "[HOP]",  # 13
        "[WHIRLPOOL]",  # 14
        "[DRY_HOP]",  # 15
        "[SPICE]",  # 16
        "[YEAST]",  # 17  (future-proof)
        # Unit tokens
        "<KG>",  # 18
        "<G>",  # 19
        "<MIN>",  # 20
        "<DAYS>",  # 21
    ]

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Train BPE on corpus WITH special tokens pre-registered
    trainer = BpeTrainer(
        vocab_size=5000,
        special_tokens=special_tokens,
        min_frequency=2,
    )

    print(f"Training tokenizer on {data_path}...")
    tokenizer.train([str(data_path)], trainer)

    # Verify special tokens are intact (not split)
    print("\n=== Special Token Verification ===")
    all_ok = True
    for st in special_tokens:
        tid = tokenizer.token_to_id(st)
        enc = tokenizer.encode(st)
        if len(enc.ids) != 1 or enc.ids[0] != tid:
            print(f"  ❌ {st:20s} → ID={tid}, encoded as {enc.tokens} (SPLIT!)")
            all_ok = False
        else:
            print(f"  ✅ {st:20s} → ID={tid}")

    if not all_ok:
        # Force-add as AddedTokens with special=True
        print("\nForce-adding split tokens...")
        for st in special_tokens:
            tokenizer.add_special_tokens([AddedToken(st, special=True)])

    # Save
    tokenizer.save(str(out_file))
    print(f"\n✅ Tokenizer saved to {out_file}")
    print(f"Vocabulary Size: {tokenizer.get_vocab_size()}")

    # Test with a real sequence
    with open(data_path, "r") as f:
        sample = f.readline().strip()

    encoded = tokenizer.encode(sample)
    print(f"\n--- Example Tokenization ({len(encoded.ids)} tokens) ---")
    print(f"Original: {sample[:120]}...")
    for tid, token in zip(encoded.ids[:25], encoded.tokens[:25]):
        print(f"  ID={tid:5d}  '{token}'")
    print(f"  ... (total {len(encoded.ids)} tokens)")

    # Verify structural tokens are single tokens
    test_cases = [
        "[START] [TARGET_ABV] 5.2",
        "[MALT] GERMAN_VIENNA 0.80 <KG>",
        "[BOIL] 60 <MIN>",
        "[HOP] CASCADE 25.0 <G>",
        "[DRY_HOP] 7 <DAYS>",
        "[END]",
    ]
    print("\n--- Structural Token Test ---")
    for tc in test_cases:
        enc = tokenizer.encode(tc)
        print(f"  '{tc}' → {enc.tokens}")


if __name__ == "__main__":
    train_brew_tokenizer()
