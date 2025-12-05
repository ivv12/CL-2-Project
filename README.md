# When Words Fall into Place: Zipf's Law and the Language of Machines

_A Comparative Analysis of Statistical Distributions in Human and LLM-Generated Texts_

## Overview

This project investigates whether large language models (LLMs) such as GPT-4, GPT-5, Claude-4, Claude-4.5, Gemini-2.5-Flash, and Gemini-2.5-Pro produce word frequency distributions that follow **Zipf's Law** in the same way human language does. By analyzing statistical distributions and linguistic properties, we quantify the "human-likeness" of LLM-generated text.

### Research Question

**Do large language models produce word frequency distributions that follow Zipf's Law in the same way human language does?** If deviations exist, can these statistical differences be systematically quantified to understand the linguistic properties and "human-likeness" of LLM-generated text?

## Key Features

- **Zipf's Law Analysis**: Compute and visualize word frequency distributions following Zipf's Law
- **Heaps' Law Analysis**: Measure vocabulary growth patterns
- **Zipf Completeness Index (ZCI)**: Quantify human-likeness of text
- **Multi-Model Comparison**: Compare 6+ LLM models against human-written essays
- **Automated Pipeline**: Process multiple text samples and generate comprehensive statistics
- **Visualization Suite**: Generate plots for individual samples and comparative analyses

## Metrics Analyzed

### 1. **Zipf's Law**
- **Slope (b)**: Ideally ~-1 for natural language
- **R² Score**: Goodness of fit to log-log distribution
- **ZCI (Zipf Completeness Index)**: Combined metric for human-likeness

### 2. **Heaps' Law**
- **β (beta)**: Vocabulary growth exponent (typically 0.4-0.8)
- **K parameter**: Scaling factor for vocabulary size
- **R² Score**: Model fit quality

## Project Structure

```
CL-2-Project/
├── data/
│   ├── human/              # Human-written essays
│   │   ├── ls.txt
│   │   ├── mc.txt
│   │   ├── mm.txt
│   │   ├── tbt.txt
│   │   └── tolentino.txt
│   └── llm/                # LLM-generated texts
│       ├── claude-4/
│       ├── claude-4.5/
│       ├── gemini-2.5-flash/
│       ├── gemini-2.5-pro/
│       ├── gpt-4/
│       └── gpt-5/
├── src/
│   ├── full_pipeline.py           # Main analysis pipeline
│   ├── tamil_full_pipeline.py     # Tamil language analysis
│   ├── heaps_least_effort.py      # Heaps' Law utilities
│   └── run_zipf_analysis.py       # Zipf analysis utilities
├── results/
│   ├── ALL_STATS.csv              # Comprehensive statistics
│   ├── *_zipf.png                 # Individual Zipf plots
│   ├── *_heaps.png                # Individual Heaps plots
│   ├── *_stats.txt                # Per-sample statistics
│   ├── zipf_slope_boxplot.png     # Comparative visualization
│   ├── heaps_beta_boxplot.png     # Comparative visualization
│   └── zci_barplot.png            # Human-likeness comparison
├── tamil_data/                     # Tamil language corpus
└── README.md
```

## Getting Started

### Prerequisites

```bash
Python 3.8+
numpy
matplotlib
pandas
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/CL-2-Project.git
cd CL-2-Project
```

2. **Install dependencies**
```bash
pip install numpy matplotlib pandas
```

### Usage

#### Run Full Analysis Pipeline

```bash
cd src
python full_pipeline.py
```

This will:
1. Process all human and LLM text samples
2. Generate individual Zipf and Heaps plots for each sample
3. Calculate comprehensive statistics (tokens, types, Zipf parameters, Heaps parameters)
4. Create comparative visualizations
5. Save all results to `results/ALL_STATS.csv`

#### Run Tamil Language Analysis

```bash
cd src
python tamil_full_pipeline.py
```

#### Individual Analysis Scripts

```bash
# Run Zipf analysis only
python run_zipf_analysis.py

# Run Heaps analysis only
python heaps_least_effort.py
```

## Output Files

### Per-Sample Outputs
- `{model}_{sample}_zipf.png`: Zipf's Law visualization (log-log plot)
- `{model}_{sample}_heaps.png`: Heaps' Law visualization (vocabulary growth)
- `{model}_{sample}_stats.txt`: Detailed statistics for the sample

### Aggregate Outputs
- `ALL_STATS.csv`: Complete statistics for all samples
- `zipf_slope_boxplot.png`: Distribution of Zipf slopes across models
- `heaps_beta_boxplot.png`: Distribution of Heaps β values across models
- `zci_barplot.png`: Mean ZCI scores per model

## Sample Results

| Model | Avg Zipf Slope (b) | Avg Heaps β | Mean ZCI |
|-------|-------------------|-------------|----------|
| Human | -0.719 | 0.808 | 0.691 |
| GPT-4 | -0.805 | 0.708 | 0.777 |
| GPT-5 | -0.727 | 0.771 | 0.706 |
| Claude-4 | -0.650 | 0.794 | 0.635 |
| Claude-4.5 | -0.720 | 0.767 | 0.704 |
| Gemini-2.5-Flash | -0.605 | 0.826 | 0.575 |
| Gemini-2.5-Pro | -0.653 | 0.813 | 0.614 |

*Note: Values shown are approximate averages from the sample data.*

## Key Findings

1. **GPT-4** shows the closest alignment to human Zipf slopes (b ≈ -0.8) and highest ZCI scores
2. **Human essays** exhibit consistent Heaps β values (~0.8), indicating natural vocabulary growth
3. **Gemini models** tend to have shallower Zipf slopes, suggesting different word usage patterns
4. **ZCI metric** effectively captures human-likeness, with GPT-4 scoring closest to human baselines

## Customization

### Adding New Text Samples

1. **Human texts**: Add `.txt` files to `data/human/`
2. **LLM texts**: Create a folder in `data/llm/` and add samples

```
data/llm/new-model/
  ├── sample1.txt
  ├── sample2.txt
  └── sample3.txt
```

3. Run the pipeline again to include new samples in analysis

### Modifying Analysis Parameters

Edit `full_pipeline.py`:

```python
# Change top-k words for Zipf analysis
zipf = compute_zipf_stats(freqs, top_k=5000)  # Default: 5000

# Change Heaps sampling step
N, V = compute_heaps(tokens, step=100)  # Default: 100
```

## Theoretical Background

### Zipf's Law
Word frequency follows: `f(r) ∝ r^(-b)` where:
- `f(r)` = frequency of word at rank `r`
- `b` ≈ -1 for natural language

### Heaps' Law
Vocabulary growth follows: `V(N) = K·N^β` where:
- `V` = vocabulary size (unique words)
- `N` = total tokens
- `β` ≈ 0.4-0.8 for natural language

### Zipf Completeness Index (ZCI)
`ZCI = R² × exp(-|b + 1|)`

Combines fit quality with ideal slope proximity to quantify human-likeness.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Research conducted as part of CL-2 course project

## Acknowledgments

- Human essay samples from various sources
- LLM text generated using GPT-4, GPT-5, Claude-4, Claude-4.5, Gemini-2.5-Flash, and Gemini-2.5-Pro
- Inspired by linguistic research on Zipf's Law and Heaps' Law

## Contact

For questions or collaboration, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes, analyzing the statistical properties of language in human and machine-generated text.ject

When Words Fall into Place: Zipf’s Law and the Language of Machines

_A Comparative Analysis of Statistical Distributions in Human and LLM-Generated Texts_

**Research Question**

Do large language models (LLMs) such as GPT, LLaMA, and Mistral produce word frequency distributions that follow Zipf’s Law in the same way human
language does? If deviations exist, can these statistical differences be systematically quantified to understand the linguistic properties and “human-likeness”
of LLM-generated text?
