import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# -------------------------------
# 1. CLEANING
# -------------------------------

def clean_text(text):
    # Remove Paul Graham notes section
    text = re.sub(r"Notes.*", "", text, flags=re.DOTALL)

    # Remove bracketed citations like [1], [2]
    text = re.sub(r"\[\d+\]", " ", text)

    # Remove punctuation except apostrophes inside words
    text = re.sub(r"[^a-zA-Z']+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()

# -------------------------------
# 2. TOKENIZATION
# -------------------------------

def tokenize(text):
    tokens = text.split()
    return [t for t in tokens if t.strip()]

# -------------------------------
# 3. FREQUENCY COUNTING
# -------------------------------

def get_freqs(tokens):
    return Counter(tokens)

# -------------------------------
# 4. ZIPF RANK–FREQUENCY + PLOT
# -------------------------------

def plot_zipf(freqs, title, outpath):
    counts = np.array(sorted(freqs.values(), reverse=True))
    ranks = np.arange(1, len(counts) + 1)

    plt.figure()
    plt.loglog(ranks, counts, marker='.', linestyle='none')
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    plt.title(title)
    plt.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()

# -------------------------------
# 5. ZIPF SLOPE, R², ZCI
# -------------------------------

def compute_zipf_stats(freqs, top_k=5000):
    counts = np.array(sorted(freqs.values(), reverse=True))[:top_k]
    ranks = np.arange(1, len(counts) + 1)

    x = np.log(ranks)
    y = np.log(counts)

    # Linear regression y = a + bx
    b, a = np.polyfit(x, y, 1)

    # R^2
    y_pred = b * x + a
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot

    # ZCI = r² * exp(-|b + 1|)
    zci = r2 * np.exp(-abs(b + 1))

    return a, b, r2, zci

# -------------------------------
# MAIN PIPELINE
# -------------------------------

def run_pipeline(name, input_path):
    print(f"\n=== PROCESSING {name.upper()} ===\n")

    raw = Path(input_path).read_text(encoding="utf-8")

    cleaned = clean_text(raw)
    tokens = tokenize(cleaned)
    freqs = get_freqs(tokens)

    # Save frequency table
    freq_out = Path(f"results/{name}_freq.tsv")
    freq_out.parent.mkdir(exist_ok=True)

    with freq_out.open("w", encoding="utf-8") as f:
        for w, c in freqs.most_common():
            f.write(f"{w}\t{c}\n")

    print(f"Saved frequency table → {freq_out}")

    # Zipf plot
    plot_path = Path(f"results/{name}_zipf_plot.png")
    plot_zipf(freqs, f"Zipf Plot — {name}", plot_path)

    print(f"Saved Zipf plot → {plot_path}")

    # Zipf stats
    a, b, r2, zci = compute_zipf_stats(freqs)
    print(f"Slope b       = {b:.4f}")
    print(f"R^2           = {r2:.4f}")
    print(f"ZCI           = {zci:.4f}")

    # Save stats
    with open(f"results/{name}_stats.txt", "w") as f:
        f.write(f"Slope (b): {b}\n")
        f.write(f"R^2: {r2}\n")
        f.write(f"ZCI: {zci}\n")

    print(f"Saved stats → results/{name}_stats.txt")

# -------------------------------
# Run for both corpora
# -------------------------------


if __name__ == "__main__":
    run_pipeline("human", "../data/human/pg_ls.txt")
    run_pipeline("llm",   "../data/llm/ls_gemini.txt")
