# LSRM: A Hybrid LLM-SBERT Approach for Mapping User Requirements to Product Functionalities in Complex Products

This repository contains the replication package for the paper "LSRM: A Hybrid LLM-SBERT Approach for Mapping User Requirements to Product Functionalities in Complex Products," submitted to RE 2025.

This package provides the datasets, source code for the LSRM method, implementations of baseline models, and experiment scripts required to reproduce the key results presented in our paper.

## Repository Structure

```
.
├── data/
│   ├── user_reviews_eng.csv          # English user reviews with ground truth
│   └── product_functionalities.csv   # Product functionality definitions
├── src/
│   ├── run_baselines_optimized.py    # Script for classic baselines (Word2Vec, USE, etc.)
│   ├── qa_zeroshot_baseline.py       # Script for QA-based Zero-Shot baseline
│   ├── ablation_exp.py               # Script for LSRM ablation studies
│   ├── llm_api_comparison.py         # Script to compare different API-based LLMs
│   └── llm_local_comparison.py       # Script to compare different local LLMs
├── models/                           # Directory for storing local models (e.g., FastText, Word2Vec)
├── requirements.txt                  # Required Python packages
└── README.md                         # This file
```

## Datasets

The `data/` directory contains the core datasets for the experiments:

*   **`user_reviews_eng.csv`**: This file contains the user reviews with the following key columns:
    - `comments_content`: The English text of the user review.
    - `index_num`: Comma-separated numerical IDs of the product functionalities relevant to the review, serving as the ground truth.

*   **`product_functionalities.csv`**: This file defines the standardized product functionalities with these essential columns:
    - `Id`: A unique numerical identifier for each functionality, which corresponds to the values in the `index_num` column of the user reviews file.
    - `Functionality Name`: The name of the product functionality.
    - `Functionality Description`: A technical description of the functionality.
    - `Functionality Experience`: A user-centric description detailing the value and usage of the functionality.

**Note on Data Origin:** The user reviews were originally collected from Chinese automotive forums and were subsequently translated into English for the experiments presented in this work.

## Setup Instructions

### 1. Environment Setup

A Python virtual environment (e.g., venv or Conda) with Python 3.10 or newer is strongly recommended to manage dependencies and ensure reproducibility.

```bash
# 1. Clone the repository
git clone [URL_OF_YOUR_GITHUB_REPOSITORY]
cd [REPOSITORY_NAME]

# 2. Create and activate a virtual environment (example using venv)
python -m venv venv_lsrm
source venv_lsrm/bin/activate  # On Windows: venv_lsrm\Scripts\activate

# 3. Install all required dependencies
pip install -r requirements.txt
```

### 2. `requirements.txt`

This file provides a list of all necessary Python packages with specific versions used in our experiments to ensure a reproducible environment. Key packages include `pandas`, `numpy`, `scikit-learn`, `torch`, `transformers`, `sentence-transformers`, `openai`, `fasttext`, `tensorflow-hub`, `gensim`, and `rank-bm25`.

**Note on Version Conflicts:** If you encounter `ImportError` or `AttributeError` related to NumPy after installation, it is likely due to version incompatibilities between `numpy`, `scipy`, and `scikit-learn`. A common solution is to create a fresh environment and install a compatible NumPy version first (e.g., `pip install "numpy<2.0"`) before installing the other packages.

### 3. Pre-trained Models and API Keys

*   **SBERT Models**: The scripts utilize Sentence-BERT models (e.g., `"all-MiniLM-L6-v2"`). These are downloaded automatically from the Hugging Face Hub by the `sentence-transformers` library on their first use. An active internet connection is required.
*   **Baseline Models (Word2Vec/FastText)**: You must download pre-trained English models separately and place them in a known local directory (e.g., `models/`). You will need to provide the path to these files via command-line arguments.
    - *Example Word2Vec Model*: Google News vectors (`GoogleNews-vectors-negative300.bin`)
    - *Example FastText Model*: Common Crawl vectors (`crawl-300d-2M-subword.bin`)
*   **Local LLMs**: For `llm_local_comparison.py`, download the desired models from the Hugging Face Hub (e.g., `google/gemma-2b-it`, `mistralai/Mistral-7B-v0.3`) and provide the local path to the model directory.
*   **LLM API Access**: For experiments that use API-based LLMs, **you must provide your API base URL and key** via command-line arguments.

## Running the Experiments

All scripts are designed to be executed from the project's root directory. The results will be saved in subdirectories within the `results/` folder (or a different directory specified by the `--output_dir` argument).

### 1. Baseline Model Comparisons

This script (`src/run_baselines_optimized.py`) evaluates classic baseline methods.

**Command (Example for Word2Vec):**
```bash
python src/run_baselines_optimized.py word2vec \
    --user_comments_csv "data/user_reviews_eng.csv" \
    --requirements_source_csv "data/product_functionalities.csv" \
    --word2vec_model_path "models/GoogleNews-vectors-negative300.bin" \
    --output_dir "results/baselines"
```
*   Replace `word2vec` with `fasttext`, `use`, or `bm25` to run other baselines.
*   Ensure the local path to the model file is correct.

### 2. QA-based Zero-Shot Baseline

This script (`src/qa_zeroshot_baseline.py`) evaluates the baseline inspired by the Question-Answering NER approach from Das et al. (RE'23).

**Command:**
```bash
python src/qa_zeroshot_baseline.py \
    --user_comments_csv "data/user_reviews_eng.csv" \
    --requirements_source_csv "data/product_functionalities.csv" \
    --llm_api_base "YOUR_LLM_API_BASE_URL" \
    --llm_api_key "YOUR_LLM_API_KEY" \
    --llm_model_name "gemini-2.0-flash" \
    --output_dir "results/baselines"
```
*   **Warning:** This script makes a large number of API calls. Use the `--num_rows_comments N` argument with a small `N` for initial testing.

### 3. LSRM Ablation Study

This script (`src/ablation_exp.py`) performs a detailed ablation study of the LSRM framework, including threshold optimization for each variant.

**Command:**
```bash
python src/ablation_exp.py \
    --user_comments_csv "data/user_reviews_eng.csv" \
    --requirements_source_csv "data/product_functionalities.csv" \
    --sbert_model_path "all-MiniLM-L6-v2" \
    --llm_api_base "YOUR_LLM_API_BASE_URL" \
    --llm_api_key "YOUR_LLM_API_KEY" \
    --llm_model_name "gemini-2.0-flash" \
    --output_dir "results/lsrm_ablation_study"
```

### 4. LLM Performance Comparison

These scripts compare the performance of different LLMs as the keyword extractor within the LSRM framework.

#### a) API-based LLMs (`src/llm_api_comparison.py`)

**Command (Example):**
```bash
# Run this command for each API-based model you wish to compare
python src/llm_api_comparison.py \
    --user_comments_csv "data/user_reviews_eng.csv" \
    --requirements_source_csv "data/product_functionalities.csv" \
    --sbert_model_path "all-MiniLM-L6-v2" \
    --llm_api_base "YOUR_LLM_API_BASE_URL" \
    --llm_api_key "YOUR_LLM_API_KEY" \
    --llm_model_name "gemini-2.0-flash" \
    --output_dir "results/llm_comparison"
```

#### b) Local LLMs (`src/llm_local_comparison.py`)

**Command (Example):**
```bash
# Run this command for each local model you wish to compare
python src/llm_local_comparison.py \
    --user_comments_csv "data/user_reviews_eng.csv" \
    --requirements_source_csv "data/product_functionalities.csv" \
    --sbert_model_path "all-MiniLM-L6-v2" \
    --local_llm_path "models/gemma-2b-it" \
    --output_dir "results/llm_comparison"
```

## Expected Output

Each script run will generate a dedicated subdirectory for its results, containing:
-   CSV files with detailed predictions on the test set.
-   CSV or TXT files summarizing the final performance metrics (Precision, Recall, F1-Score, Accuracy).
-   PNG plots visualizing the results, such as performance vs. thresholds where applicable.

## Citation

If you use this work, please cite our paper:

```
Zhiwei Zhang, et al. "LSRM: A Hybrid LLM-SBERT Approach for Mapping User Requirements to Product Functionalities in Complex Products." In Proceedings of the IEEE International Requirements Engineering Conference (RE), 2025. 
```

## Contact

For questions regarding this replication package, please contact me at `zhangzhiwei1019@link.cuhk.edu.hk`.
