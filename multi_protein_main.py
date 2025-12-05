import torch
import esm
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hooks import PatchManager

# ==========================================
# 0. The Dataset (Diverse Folds)
# ==========================================
PROTEINS = [
    ("Ubiquitin", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"),
    ("Protein G", "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"), 
    ("Protein A", "MNAAQHDEAQQNAFYQVLNMPNLNADQRNGFIQSLKDDPSQSANVLGEAQKLNDSQAPK"), 
    ("SH3 Domain", "DETGKELVLALYDYQEKSPREVTMKKGDILTLLNSTNKDWWKVEVNDRQGFVPAAYVKKLD"), 
    ("Homeodomain", "RRRKRTAEREAELQKIVSEPGDSVKKKEGERLKQLYIEQSNKNRAIKRLEIQ"),
    ("Zinc Finger", "YKCGLCERSFVEKSALSRHQKRHTGEKPYK"), 
    ("WW Domain", "PLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPR"), 
    ("Trp-Cage", "DAYAQWLKDGGPSSGRPPPS"), 
]

# ==========================================
# 1. Helpers
# ==========================================

def get_best_contact(model, alphabet, sequence, device):
    """Finds the strongest long-range contact."""
    batch_converter = alphabet.get_batch_converter()
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = model(batch_tokens, return_contacts=True)
        contacts = results["contacts"][0]

    L = contacts.shape[0]
    candidates = []
    
    # 1. Scan for contacts
    min_sep = 10 # Enforce strict long-range to find "Deep" circuits
    for i in range(1, L-1):
        for j in range(i + min_sep, L-1): 
            score = contacts[i, j].item()
            if score > 0.6: # Lowered slightly to capture more candidates
                candidates.append(((i, j), score, j - i))
    
    if not candidates:
        return None, None
        
    # 2. Sort by DISTANCE first, then Score.
    # This ensures we pick the hardest structural features, which usually
    # require the Layer 4 "Driver" mechanism.
    candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
    
    return candidates[0][0], candidates[0][1]

def create_corrupted_seq(sequence, t_i, t_j):
    # Convert tensor index to string index
    i, j = t_i - 1, t_j - 1
    span_start, span_end = i + 1, j
    
    if span_end <= span_start: return sequence
    
    s = list(sequence)
    mid = s[span_start:span_end]
    random.shuffle(mid)
    s[span_start:span_end] = mid
    return "".join(s)

# ==========================================
# 2. The Core Experiment (MLP Sweep)
# ==========================================

def run_experiment(model, alphabet, device, name, seq):
    print(f"Processing {name} (Len: {len(seq)})...")
    
    # 1. Find Contact
    result = get_best_contact(model, alphabet, seq, device)
    
    # --- FIX: Handle None safely ---
    if result[0] is None:
        print(f"  -> Skipping (No strong long-range contacts > 10 residues)")
        return None
    
    (t_i, t_j), clean_score = result
    print(f"  -> Target: ({t_i}, {t_j}) Dist: {t_j - t_i} Score: {clean_score:.2f}")

    # 2. Corrupt
    corr_seq = create_corrupted_seq(seq, t_i, t_j)
    
    batch_converter = alphabet.get_batch_converter()
    _, _, clean_toks = batch_converter([("c", seq)])
    _, _, corr_toks = batch_converter([("d", corr_seq)])
    clean_dict = {"tokens": clean_toks.to(device)}
    corr_dict = {"tokens": corr_toks.to(device)}

    pm = PatchManager(model, device)
    
    # 3. Get Baselines
    with torch.no_grad():
        with pm.recording():
            model(clean_dict["tokens"], return_contacts=True)
        
        res_corr = model(corr_dict["tokens"], return_contacts=True)
        corr_score = res_corr["contacts"][0, t_i, t_j].item()

    denom = clean_score - corr_score
    if denom < 0.1:
        print(f"  -> Skipping (Signal differential too weak: {denom:.2f})")
        return None
        
    # 4. Sweep MLP Layers
    layer_recoveries = []
    num_layers = len(model.layers)
    
    for layer_idx in range(num_layers):
        pm.set_target(mlp_layers={layer_idx})
        with torch.no_grad():
            with pm.patching():
                res = model(corr_dict["tokens"], return_contacts=True)
                patched_score = res["contacts"][0, t_i, t_j].item()
        
        recovery = (patched_score - corr_score) / denom
        layer_recoveries.append(recovery)
        
    print(f"  -> L4 Recovery: {layer_recoveries[4]:.2f}")
    return layer_recoveries

# ==========================================
# 3. Aggregation & Visualization
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading ESM-2...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().to(device)
    
    results = [] 
    
    for name, seq in PROTEINS:
        res = run_experiment(model, alphabet, device, name, seq)
        if res:
            results.append(res)
            
    if not results:
        print("No valid results collected.")
        return

    # Convert to Numpy
    data = np.array(results)
    num_layers = data.shape[1]
    
    # Print Stats
    print("\n" + "="*30)
    print(f"AGGREGATE RESULTS (N={len(results)})")
    print("="*30)
    print("Layer | Mean Rec | Std Dev")
    print("-" * 25)
    for l in range(num_layers):
        print(f"MLP {l} | {data[:, l].mean():7.2f}  | {data[:, l].std():5.2f}")
        
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Create Boxplot
    sns.boxplot(data=data, color="lightblue")
    sns.swarmplot(data=data, color=".25")
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axhline(1, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.title(f"Mechanism Consistency Across {len(results)} Proteins (ESM-2 8M)")
    plt.ylabel("Recovery Score (1.0 = Full Restoration)")
    plt.xlabel("MLP Layer Index")
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()