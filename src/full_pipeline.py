import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import pandas as pd

# ======================================================
# 1. CLEANING + TOKENIZATION
# ======================================================

def clean_text(text):
    # Remove Gutenberg-style notes if present
    text = re.sub(r"Notes.*", "", text, flags=re.DOTALL)

    # Remove bracketed numbers e.g., [23]
    text = re.sub(r"\[\d+\]", " ", text)

    # Keep Unicode word chars + apostrophes
    text = re.sub(r"[^\w']+", " ", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokenize(text):
    return text.split()


def get_freqs(tokens):
    return Counter(tokens)


# ======================================================
# 2. ZIPF ANALYSIS
# ======================================================

def compute_zipf_stats(freqs, top_k=5000):
    counts = np.array(sorted(freqs.values(), reverse=True))[:top_k]

    if len(counts) < 2:
        return None

    ranks = np.arange(1, len(counts) + 1)

    x = np.log(ranks)
    y = np.log(counts)

    b, a = np.polyfit(x, y, 1)

    y_pred = a + b * x
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    # Zipf completeness index
    zci = r2 * np.exp(-abs(b + 1))

    return a, b, r2, zci


# ======================================================
# 3. HEAPS ANALYSIS
# ======================================================

def compute_heaps(tokens, step=100):
    vocab = set()
    Ns, Vs = [], []

    for i in range(0, len(tokens), step):
        batch = tokens[i:i + step]
        for t in batch:
            vocab.add(t)
        Ns.append(i + len(batch))
        Vs.append(len(vocab))

    return np.array(Ns), np.array(Vs)


def fit_heaps(N, V):
    if len(N) < 2:
        return None

    logN, logV = np.log(N), np.log(V)
    beta, logK = np.polyfit(logN, logV, 1)
    K = np.exp(logK)

    V_pred = K * (N ** beta)
    ss_res = ((V - V_pred) ** 2).sum()
    ss_tot = ((V - V.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    return K, beta, r2


# ======================================================
# 4. PLOTS FOR EACH FILE
# ======================================================

def save_zipf_plot(freqs, model, sample, outdir):
    counts = np.array(sorted(freqs.values(), reverse=True))
    ranks = np.arange(1, len(counts) + 1)

    plt.figure(figsize=(6,4))
    plt.loglog(ranks, counts)
    plt.title(f"Zipf Plot — {model} {sample}")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outdir / f"{model}_{sample}_zipf.png")
    plt.close()


def save_heaps_plot(N, V, model, sample, outdir):
    plt.figure(figsize=(6,4))
    plt.plot(N, V)
    plt.title(f"Heaps Plot — {model} {sample}")
    plt.xlabel("Tokens (N)")
    plt.ylabel("Vocabulary Size (V)")
    plt.tight_layout()
    plt.savefig(outdir / f"{model}_{sample}_heaps.png")
    plt.close()


def save_stats_text(stats, outpath):
    with open(outpath, "w", encoding="utf-8") as f:
        for k,v in stats.items():
            f.write(f"{k}: {v}\n")


# ======================================================
# 5. PROCESS ONE FILE
# ======================================================

def process_file(model, version, sample, filepath, outdir):
    raw = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(raw)
    tokens = tokenize(cleaned)

    if len(tokens) < 40:
        print(f"⚠ SKIPPED (too few tokens): {filepath}")
        return None

    freqs = get_freqs(tokens)

    zipf = compute_zipf_stats(freqs)
    if zipf is None:
        print(f"⚠ SKIPPED (invalid Zipf data): {filepath}")
        return None
    a, b, r2, zci = zipf

    N, V = compute_heaps(tokens)
    heaps = fit_heaps(N, V)
    if heaps is None:
        print(f"⚠ SKIPPED (invalid Heaps data): {filepath}")
        return None
    K, beta, h_r2 = heaps

    # ---------- SAVE PER-FILE PLOTS ----------
    save_zipf_plot(freqs, model, sample, outdir)
    save_heaps_plot(N, V, model, sample, outdir)

    # ---------- SAVE PER-FILE STATS ----------
    stats = {
        "model": model,
        "version": version,
        "sample": sample,
        "tokens": len(tokens),
        "types": len(freqs),
        "zipf_b": b,
        "zipf_r2": r2,
        "zci": zci,
        "heaps_beta": beta,
        "heaps_r2": h_r2,
    }
    save_stats_text(stats, outdir / f"{model}_{sample}_stats.txt")

    return stats


# ======================================================
# 6. WALK THROUGH ALL DATA
# ======================================================

def collect_all_stats():
    ROOT = Path(__file__).resolve().parent.parent
    HUMAN_DIR = ROOT / "data/human"
    LLM_DIR = ROOT / "data/llm"
    OUT = ROOT / "results"
    OUT.mkdir(exist_ok=True)

    all_rows = []

    # ---- HUMAN ----
    for file in HUMAN_DIR.glob("*.txt"):
        stats = process_file("human", "essay", file.stem, file, OUT)
        if stats: 
            all_rows.append(stats)
            print(f"Processed HUMAN: {file.name}")

    # ---- LLM MODELS ----
    for model_folder in LLM_DIR.iterdir():
        if not model_folder.is_dir():
            continue

        model_name = model_folder.name

        for file in model_folder.glob("*.txt"):
            stats = process_file(model_name, model_name, file.stem, file, OUT)
            if stats:
                all_rows.append(stats)
                print(f"Processed LLM: {model_name} — {file.name}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "ALL_STATS.csv", index=False)
    print("\n✓ Saved FULL combined stats → results/ALL_STATS.csv\n")
    return df


# ======================================================
# 7. GLOBAL COMPARISON PLOTS
# ======================================================

def comparison_plots(df):
    OUT = Path(__file__).resolve().parent.parent / "results"

    # ZIPF slope boxplot
    plt.figure(figsize=(10,5))
    df.boxplot(column="zipf_b", by="model", rot=45)
    plt.title("Zipf Slope b Per Model")
    plt.suptitle("")
    plt.ylabel("Slope b")
    plt.tight_layout()
    plt.savefig(OUT / "zipf_slope_boxplot.png")
    plt.close()

    # HEAPS beta boxplot
    plt.figure(figsize=(10,5))
    df.boxplot(column="heaps_beta", by="model", rot=45)
    plt.title("Heaps β Per Model")
    plt.suptitle("")
    plt.ylabel("β")
    plt.tight_layout()
    plt.savefig(OUT / "heaps_beta_boxplot.png")
    plt.close()

    # ZCI barplot
    plt.figure(figsize=(10,5))
    df.groupby("model")["zci"].mean().sort_values().plot(kind="bar")
    plt.title("Mean ZCI Per Model")
    plt.ylabel("ZCI (higher = more human-like)")
    plt.tight_layout()
    plt.savefig(OUT / "zci_barplot.png")
    plt.close()

    print("✓ Saved all comparison plots\n")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    df = collect_all_stats()
    comparison_plots(df)
    print("✓ FULL ANALYSIS COMPLETE\n")
