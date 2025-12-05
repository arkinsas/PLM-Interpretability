import torch
import esm
import random
from contact_patching import run_contact_sweep 
from hooks import PatchManager


def get_best_long_range_contact(model, alphabet, sequence, device):
    """
    1. Must have probability > 0.8 (Strong Clean Signal)
    2. Sorts by separation distance (Hardest task = Strongest Causal Link)
    """
    print(f"Scanning contacts for sequence length {len(sequence)}...")
    batch_converter = alphabet.get_batch_converter()
    data = [("protein1", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = model(batch_tokens, return_contacts=True)
        contacts = results["contacts"][0]  # Shape: (L, L)

    L = contacts.shape[0]
    candidates = []
    
    # Collect valid strong contacts (min separation 10)
    for i in range(1, L-1):
        for j in range(i + 10, L-1): 
            score = contacts[i, j].item()
            if score > 0.8: 
                candidates.append(((i, j), score, j - i))
    
    if not candidates:
        return None, None
        
    # Sort by Separation Distance (Descending), then Score (Descending)
    candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
    
    best_contact, best_score, dist = candidates[0]
    print(f"Selected Target: Indices {best_contact} | Separation: {dist} | Score: {best_score:.4f}")
    return best_contact, best_score

def create_corrupted_shuffled(sequence, tensor_idx_i, tensor_idx_j):
    """
    Shuffles the span between contacts to break the interaction context.
    """
    i = tensor_idx_i - 1
    j = tensor_idx_j - 1
    
    span_start = i + 1
    span_end = j
    
    if span_end <= span_start:
        return sequence
        
    seq_list = list(sequence)
    middle_segment = seq_list[span_start:span_end]
    random.shuffle(middle_segment)
    seq_list[span_start:span_end] = middle_segment
    
    return "".join(seq_list)


def verify_cumulative_recovery(model, alphabet, device, pair, top_heads, num_cumulative=10):
    """
    Patches the Top 1...N heads simultaneously.
    """
    print("\n" + "="*40)
    print(f"Running Cumulative Patching on Top {num_cumulative} Heads")
    print("="*40)

    clean_seq, corrupt_seq, (t_i, t_j) = pair
    batch_converter = alphabet.get_batch_converter()
    
    # Prepare Tokens
    _, _, clean_toks = batch_converter([("c", clean_seq)])
    _, _, corr_toks = batch_converter([("d", corrupt_seq)])
    clean_dict = {"tokens": clean_toks.to(device)}
    corr_dict = {"tokens": corr_toks.to(device)}

    pm = PatchManager(model, device)

    # 1. Record Clean
    with torch.no_grad():
        with pm.recording():
            res_clean = model(clean_dict["tokens"], return_contacts=True)
            clean_prob = res_clean["contacts"][0, t_i, t_j].item()
    
    # 2. Record Corrupt Baseline
    with torch.no_grad():
        res_corr = model(corr_dict["tokens"], return_contacts=True)
        corr_prob = res_corr["contacts"][0, t_i, t_j].item()

    print(f"Baseline Clean:   {clean_prob:.4f}")
    print(f"Baseline Corrupt: {corr_prob:.4f}")
    print("-" * 20)

    for k in range(1, num_cumulative + 1):
        current_heads = top_heads[:k]
        target_set = {(h['layer'], h['head']) for h in current_heads}
        
        pm.set_target(heads=target_set)

        with torch.no_grad():
            with pm.patching():
                res_patched = model(corr_dict["tokens"], return_contacts=True)
                patched_prob = res_patched["contacts"][0, t_i, t_j].item()

        denom = clean_prob - corr_prob
        if abs(denom) < 1e-6: denom = 1e-6 
        recovery = (patched_prob - corr_prob) / denom
        
        print(f"Top {k} Heads: Score {patched_prob:.4f} | Recovery {recovery:.2f}")

def sweep_mlps(model, alphabet, device, pair):
    """
    Sweeps through MLP layers to check for Signal Driving or Suppression.
    """
    print("\n" + "="*40)
    print("Running MLP Sweep (Feed Forward Layers)")
    print("="*40)

    clean_seq, corrupt_seq, (t_i, t_j) = pair
    batch_converter = alphabet.get_batch_converter()
    
    _, _, clean_toks = batch_converter([("c", clean_seq)])
    _, _, corr_toks = batch_converter([("d", corrupt_seq)])
    clean_dict = {"tokens": clean_toks.to(device)}
    corr_dict = {"tokens": corr_toks.to(device)}

    pm = PatchManager(model, device)
    
    # Baselines
    with torch.no_grad():
        with pm.recording():
            res_clean = model(clean_dict["tokens"], return_contacts=True)
            clean_prob = res_clean["contacts"][0, t_i, t_j].item()
            
    with torch.no_grad():
        res_corr = model(corr_dict["tokens"], return_contacts=True)
        corr_prob = res_corr["contacts"][0, t_i, t_j].item()
        
    denom = clean_prob - corr_prob
    if abs(denom) < 1e-9: denom = 1e-9

    # Sweep Layers
    for layer_idx in range(len(model.layers)):
        # Target only the MLP of this layer
        pm.set_target(mlp_layers={layer_idx})
        
        with torch.no_grad():
            with pm.patching():
                res = model(corr_dict["tokens"], return_contacts=True)
                patched_prob = res["contacts"][0, t_i, t_j].item()
        
        recovery = (patched_prob - corr_prob) / denom
        print(f"MLP Layer {layer_idx}: Recovery {recovery:.2f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().to(device)

    # UBIQUITIN Sequence
    clean_seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    
    # 1. Find the best contact
    (t_i, t_j), score = get_best_long_range_contact(model, alphabet, clean_seq, device)
    
    if t_i is None:
        print("No strong contacts found.")
        return

    # 2. Corrupt
    corrupted_seq = create_corrupted_shuffled(clean_seq, t_i, t_j)
    pairs = [(clean_seq, corrupted_seq, (t_i, t_j))]

    # 3. Single Head Sweep
    print("Running activation patching sweep (Single Heads)...")
    res = run_contact_sweep(model, alphabet, device, pairs)

    pair_res = res["pairs"][0]
    clean_score = pair_res['clean_metric']
    corr_score = pair_res['corrupted_metric']
    
    print("\nSingle Head Results:")
    print(f"Clean:   {clean_score:.4f}")
    print(f"Corrupt: {corr_score:.4f}")
    print(f"Delta:   {clean_score - corr_score:.4f}")
    print("-" * 30)
    
    top_heads = sorted(pair_res["head_results"], key=lambda x: x["recovery"], reverse=True)
    for h in top_heads[:5]:
        print(f"Layer {h['layer']} Head {h['head']}: Recovery {h['recovery']:.2f}")
    
    # 4. Cumulative Heads Verification
    verify_cumulative_recovery(model, alphabet, device, pairs[0], top_heads, num_cumulative=10)
    
    # 5. MLP Sweep (New)
    sweep_mlps(model, alphabet, device, pairs[0])

if __name__ == "__main__":
    main()
