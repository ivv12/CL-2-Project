import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# -------------------------
# CLEAN + TOKENIZE
# -------------------------

def clean_text(text):
    # Remove PG notes
    text = re.sub(r"Notes.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"[^a-zA-Z']+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def tokenize(text):
    return text.split()

# -------------------------
# HEAPS' LAW ANALYSIS
# -------------------------

def compute_heaps(tokens, step=100):
    vocab = set()
    Ns = []
    Vs = []

    for i in range(0, len(tokens), step):
        batch = tokens[i:i+step]
        for t in batch:
            vocab.add(t)
        Ns.append(i + len(batch))
        Vs.append(len(vocab))

    return np.array(Ns), np.array(Vs)

def fit_heaps(N, V):
    logN = np.log(N)
    logV = np.log(V)

    # Fit log(V) = log(K) + beta * log(N)
    beta, logK = np.polyfit(logN, logV, 1)
    K = np.exp(logK)

    # R²
    V_pred = K * (N ** beta)
    ss_res = np.sum((V - V_pred)**2)
    ss_tot = np.sum((V - np.mean(V))**2)
    r2 = 1 - ss_res / ss_tot

    return K, beta, r2

# -------------------------
# PLOTTING
# -------------------------

def plot_heaps(N, V, K, beta, name):
    plt.figure()
    plt.loglog(N, V, '.', label="Observed")
    plt.loglog(N, K * (N ** beta), label=f"Fit: V = {K:.2f}·N^{beta:.2f}")
    plt.xlabel("Tokens N (log)")
    plt.ylabel("Vocabulary Size V (log)")
    plt.title(f"Heaps' Law – {name}")
    plt.legend()
    plt.tight_layout()

    out = Path(f"results/{name}_heaps_plot.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out)
    plt.close()

    print(f"Saved Heaps plot → {out}")

# -------------------------
# PIPELINE
# -------------------------

def run_heaps(name, path):
    print(f"\n=== HEAPS ANALYSIS FOR {name.upper()} ===")

    raw = Path(path).read_text(encoding="utf-8")
    cleaned = clean_text(raw)
    tokens = tokenize(cleaned)

    # Compute curves
    N, V = compute_heaps(tokens)

    # Fit Heaps law
    K, beta, r2 = fit_heaps(N, V)

    # Save stats
    stats_path = Path(f"results/{name}_heaps_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"K: {K}\n")
        f.write(f"beta: {beta}\n")
        f.write(f"R^2: {r2}\n")

    print(f"K     = {K:.4f}")
    print(f"beta  = {beta:.4f}")
    print(f"R^2   = {r2:.4f}")
    print(f"Saved Heaps stats → {stats_path}")

    # Plot
    plot_heaps(N, V, K, beta, name)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent

    run_heaps("human", ROOT/"data/human/pg_ls.txt")
    run_heaps("llm",   ROOT/"data/llm/ls_gpt5.txt")
