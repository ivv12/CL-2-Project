# tamil_full_pipeline.py
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import pandas as pd

# ======================================================
# CONFIG
# ======================================================
TAMIL_START = 0x0B80
TAMIL_END = 0x0BFF

# ======================================================
# 1. CLEANING + TOKENIZATION
# ======================================================

def clean_text(text):
    text = re.sub(r"Notes.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"[^A-Za-z0-9\u0B80-\u0BFF']+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text):
    return text.split()


def contains_tamil_char(token):
    return any(TAMIL_START <= ord(ch) <= TAMIL_END for ch in token)


def all_chars_allowed(token):
    for ch in token:
        code = ord(ch)
        if (TAMIL_START <= code <= TAMIL_END) or ch.isalnum() or ch == "'":
            continue
        return False
    return True


def is_valid_tamil_token(token):
    if not token:
        return False
    if not contains_tamil_char(token):
        return False
    return all_chars_allowed(token)


def get_freqs(tokens):
    return Counter(tokens)


# ======================================================
# ZIPF + HEAPS
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

    zci = r2 * np.exp(-abs(b + 1))

    return a, b, r2, zci


def compute_heaps(tokens, step=100):
    vocab = set()
    Ns, Vs = [], []

    for i in range(0, len(tokens), step):
        batch = tokens[i:i+step]
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

    V_pred = K * (N**beta)
    ss_res = ((V - V_pred)**2).sum()
    ss_tot = ((V - V.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot

    return K, beta, r2


# ======================================================
# PLOTS
# ======================================================

def save_zipf_plot(freqs, model, sample, outdir):
    counts = np.array(sorted(freqs.values(), reverse=True))
    if len(counts) == 0:
        return
    ranks = np.arange(1, len(counts)+1)

    plt.figure(figsize=(6,4))
    plt.loglog(ranks, counts)
    plt.title(f"Tamil Zipf — {model} {sample}")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outdir / f"{model}_{sample}_zipf.png")
    plt.close()


def save_heaps_plot(N, V, model, sample, outdir):
    plt.figure(figsize=(6,4))
    plt.plot(N, V)
    plt.title(f"Tamil Heaps — {model} {sample}")
    plt.xlabel("Tokens")
    plt.ylabel("Vocabulary Size")
    plt.tight_layout()
    plt.savefig(outdir / f"{model}_{sample}_heaps.png")
    plt.close()


# ======================================================
# PROCESS ONE FILE
# ======================================================

def process_file(model, version, sample, filepath, outdir):
    raw = filepath.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(raw)
    tokens = [t for t in tokenize(cleaned) if is_valid_tamil_token(t)]

    if len(tokens) < 40:
        print(f"⚠ SKIPPED (too few Tamil tokens): {filepath}")
        return None

    freqs = get_freqs(tokens)

    zipf = compute_zipf_stats(freqs)
    if zipf is None:
        return None
    a, b, r2, zci = zipf

    N, V = compute_heaps(tokens)
    heaps = fit_heaps(N, V)
    if heaps is None:
        return None
    K, beta, h_r2 = heaps

    # save per-file plots
    save_zipf_plot(freqs, model, sample, outdir)
    save_heaps_plot(N, V, model, sample, outdir)

    stats = dict(
        model=model,
        version=version,
        sample=sample,
        tokens=len(tokens),
        types=len(freqs),
        zipf_b=float(b),
        zipf_r2=float(r2),
        zci=float(zci),
        heaps_beta=float(beta),
        heaps_r2=float(h_r2),
    )

    with open(outdir / f"{model}_{sample}_stats.txt", "w", encoding="utf-8") as f:
        for k,v in stats.items():
            f.write(f"{k}: {v}\n")

    return stats


# ======================================================
# WALK THROUGH FOLDERS (FIXED)
# ======================================================

def collect_all_stats():
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "tamil_data"
    HUMAN_DIR = DATA_DIR / "human"
    LLM_DIR = DATA_DIR / "llm"

    OUT_ROOT = ROOT / "results" / "tamil"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # --- HUMAN ---
    for file in sorted(HUMAN_DIR.glob("*.txt")):
        stats = process_file("human", "essay", file.stem, file, OUT_ROOT)
        if stats:
            all_rows.append(stats)
            print(f"Processed HUMAN: {file.name}")

    # --- LLM MODELS ---
    for model_folder in sorted(LLM_DIR.iterdir()):
        if not model_folder.is_dir():
            continue

        model_name = model_folder.name

        for file in sorted(model_folder.glob("*.txt")):
            stats = process_file(model_name, model_name, file.stem, file, OUT_ROOT)
            if stats:
                all_rows.append(stats)
                print(f"Processed LLM: {model_name} — {file.name}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_ROOT / "TAMIL_STATS.csv", index=False)
    print("✓ Saved FULL Tamil stats →", OUT_ROOT / "TAMIL_STATS.csv")

    return df, OUT_ROOT


# ======================================================
# GLOBAL PLOTS
# ======================================================

def comparison_plots(df, out_root):
    plt.figure(figsize=(10,5))
    df.boxplot(column="zipf_b", by="model", rot=45)
    plt.title("Zipf Slope b Per Model (Tamil)")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(out_root / "zipf_slope_boxplot_tamil.png")
    plt.close()

    plt.figure(figsize=(10,5))
    df.boxplot(column="heaps_beta", by="model", rot=45)
    plt.title("Heaps β Per Model (Tamil)")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(out_root / "heaps_beta_boxplot_tamil.png")
    plt.close()

    plt.figure(figsize=(10,5))
    df.groupby("model")["zci"].mean().sort_values().plot(kind="bar")
    plt.title("Mean ZCI Per Model (Tamil)")
    plt.tight_layout()
    plt.savefig(out_root / "zci_barplot_tamil.png")
    plt.close()

    print("✓ Saved Tamil comparison plots")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    df, outroot = collect_all_stats()
    if df.empty:
        print("⚠ No Tamil data processed.")
    else:
        comparison_plots(df, outroot)
        print("✓ TAMIL ANALYSIS COMPLETE")
