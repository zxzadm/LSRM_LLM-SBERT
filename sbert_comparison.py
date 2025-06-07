import pandas as pd
import numpy as np
import re
import os
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle
import time

# --- Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# --- Data Handling & Keyword Extraction Functions ---
def read_and_split_sentences(csv_file, content_column='comments_content', nrows=None):
    df = pd.read_csv(csv_file, nrows=nrows) if nrows else pd.read_csv(csv_file)
    comments = df[content_column].fillna('').astype(str).tolist()
    sentences = [[s.strip() for s in re.split(r'[.!?\n]+', c) if s.strip()] for c in comments]
    return sentences, df

def load_product_functionalities(csv_file, id_col='Id', name_col='Functionality Name', desc_col='Functionality Description', exp_col='Functionality Experience'):
    df = pd.read_csv(csv_file)
    ids = df[id_col].tolist()
    names = df[name_col].astype(str).tolist()
    descriptions = df[desc_col].astype(str).fillna('').tolist()
    experiences = df[exp_col].astype(str).fillna('').tolist() if exp_col in df.columns else [''] * len(df)
    combined_descs = [f"{desc} {exp}".strip() for desc, exp in zip(descriptions, experiences)]
    name_to_id_map = {name: fid for name, fid in zip(names, ids)}
    all_func_ids_set = set(ids)
    return names, combined_descs, name_to_id_map, all_func_ids_set

def extract_keywords_via_api(text, llm_client, model_name):
    # English prompt for keyword extraction
    system_prompt = "You are an expert in product requirement analysis for new energy vehicles. Your task is to accurately extract keywords reflecting specific functional requirements from user reviews."
    user_prompt = f"""Analyze the following user review sentence: "{text}"
Please follow this framework:
1. Identify specific car functions, performance aspects, or user experience points mentioned.
2. Convert these points into concise functional requirement keywords.
3. Extract only genuine requirements explicitly stated or strongly implied.
Keywords should be specific and precise (e.g., "automatic parking" is better than "parking").
Example:
Review: "The seats get cold in winter; I wish it had a heating function. Also, the navigation is often inaccurate."
Keywords: seat heating, navigation accuracy
Output a comma-separated list of keywords directly, with no other explanation:"""
    
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        completion = llm_client.chat.completions.create(model=model_name, messages=messages, temperature=0.1, max_tokens=100)
        keywords_str = completion.choices[0].message.content
        return [kw.strip() for kw in keywords_str.split(',')] if keywords_str else ["no_keywords"]
    except Exception as e:
        print(f"API call error for text '{text[:40]}...': {str(e)}")
        return ["api_error"]

def get_all_keywords_with_cache(sentences_per_comment, llm_client, model_name, cache_file='extracted_keywords.pkl'):
    if os.path.exists(cache_file):
        print(f"Loading cached keywords from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    keywords_dict = {}
    print(f"\nExtracting keywords using {model_name} (will be cached)...")
    for i, sentences in enumerate(tqdm(sentences_per_comment, desc="Extracting Keywords")):
        for j, sentence in enumerate(sentences):
            key = f"{i}_{j}"
            keywords = extract_keywords_via_api(sentence, llm_client, model_name)
            keywords_dict[key] = keywords
            time.sleep(0.05) # Small delay to avoid hitting API rate limits
    
    print(f"Saving extracted keywords to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(keywords_dict, f)
    return keywords_dict

# --- Core Logic Functions ---
def generate_user_embeddings(sentences_per_comment, sbert_model, keywords_dict):
    user_embeddings = []
    sbert_model_name = os.path.basename(sbert_model.get_tokenizer().name_or_path) # Get model name for progress bar
    for i, sentences in enumerate(tqdm(sentences_per_comment, desc=f"User Embeddings ({sbert_model_name})")):
        comment_sentence_embs = []
        if not sentences:
            user_embeddings.append(np.array([]))
            continue
        original_sent_embs = sbert_model.encode(sentences, show_progress_bar=False)
        for j, sentence in enumerate(sentences):
            key = f"{i}_{j}"
            keywords = keywords_dict.get(key, ["no_keywords"])
            keyword_emb = sbert_model.encode([' '.join(keywords)])[0]
            combined_emb = np.concatenate((original_sent_embs[j], keyword_emb))
            comment_sentence_embs.append(combined_emb)
        user_embeddings.append(np.array(comment_sentence_embs))
    return user_embeddings

def generate_feature_embeddings(func_names, func_descs, sbert_model):
    sbert_model_name = os.path.basename(sbert_model.get_tokenizer().name_or_path)
    print(f"\nGenerating feature embeddings with {sbert_model_name}...")
    name_embs = sbert_model.encode(func_names, show_progress_bar=True)
    desc_embs = sbert_model.encode(func_descs, show_progress_bar=True)
    return [np.concatenate((n, d)) for n, d in zip(name_embs, desc_embs)]

# --- Evaluation and Visualization Functions --- (These can be reused from other scripts)
def map_and_predict(user_embeddings, feature_embeddings, feature_names, threshold):
    # Same logic as in other scripts
    predictions = []
    for comment_embs in user_embeddings:
        mapped_features = set()
        if comment_embs.size > 0:
            for sent_emb in comment_embs:
                similarities = util.cos_sim(sent_emb, feature_embeddings)[0]
                for i, score in enumerate(similarities):
                    if score >= threshold:
                        mapped_features.add(feature_names[i])
        predictions.append(", ".join(sorted(list(mapped_features))))
    return predictions

def evaluate(true_labels_sets, predicted_labels_sets, all_func_ids_set):
    # Same logic as in other scripts
    precisions, recalls, f1s = [], [], []
    y_true_flat, y_pred_flat = [], []
    feature_list = sorted(list(all_func_ids_set))
    for true_set, pred_set in zip(true_labels_sets, predicted_labels_sets):
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
        y_true_flat.extend([1 if fid in true_set else 0 for fid in feature_list])
        y_pred_flat.extend([1 if fid in pred_set else 0 for fid in feature_list])
    return {"precision": np.mean(precisions), "recall": np.mean(recalls), "f1": np.mean(f1s), "accuracy": accuracy_score(y_true_flat, y_pred_flat)}

def visualize_sbert_comparison(all_results, output_dir):
    # Same visualization logic as in the provided script
    models = list(all_results.keys())
    metrics_data = {metric: [all_results[m][metric] for m in models] for metric in ['precision', 'recall', 'f1', 'accuracy']}
    df = pd.DataFrame(metrics_data, index=models)
    
    df.plot(kind='bar', figsize=(12, 7))
    plt.title('Performance Comparison of Different SBERT Models in LSRM')
    plt.ylabel('Score')
    plt.xlabel('SBERT Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sbert_comparison_bar_chart.png'))
    plt.close()
    print(f"Saved SBERT comparison plot to {os.path.join(output_dir, 'sbert_comparison_bar_chart.png')}")

# --- Main Execution ---
def main(args):
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Load Data and Keywords ---
    user_sentences, user_df = read_and_split_sentences(args.user_comments_csv, args.user_content_col, args.num_rows_comments)
    func_names, func_descs, name_to_id_map, all_func_ids_set = load_product_functionalities(args.requirements_source_csv, args.req_id_col, args.req_name_col, args.req_desc_col, args.req_exp_col)
    true_labels = [{name_to_id_map[n] for n in p.split(', ') if n and n in name_to_id_map} for p in user_df[args.user_true_label_col].fillna('')]

    llm_client = OpenAI(base_url=args.llm_api_base, api_key=args.llm_api_key)
    keywords_dict = get_all_keywords_with_cache(user_sentences, llm_client, args.llm_model_name, args.keyword_cache_file)
    
    # --- Iterate Over SBERT Models ---
    all_sbert_results = {}
    for sbert_model_path in args.sbert_models:
        sbert_model_name = os.path.basename(sbert_model_path) if os.path.isdir(sbert_model_path) else sbert_model_path
        print(f"\n===== Evaluating SBERT Model: {sbert_model_name} =====")
        sbert_model = SentenceTransformer(sbert_model_path)
        
        # Generate embeddings with the current SBERT model
        user_embeddings = generate_user_embeddings(user_sentences, sbert_model, keywords_dict)
        feature_embeddings = generate_feature_embeddings(func_names, func_descs, sbert_model)
        
        # Map and predict at the fixed threshold
        predicted_feature_names = map_and_predict(user_embeddings, feature_embeddings, func_names, args.threshold)
        predicted_labels = [{name_to_id_map[n] for n in p.split(', ') if n and n in name_to_id_map} for p in predicted_feature_names]

        # Evaluate performance
        performance = evaluate(true_labels, predicted_labels, all_func_ids_set)
        all_sbert_results[sbert_model_name] = performance
        print(f"Performance for {sbert_model_name}: {performance}")

    # --- Save and Visualize Final Results ---
    results_df = pd.DataFrame.from_dict(all_sbert_results, orient='index')
    results_df.to_csv(os.path.join(output_dir, "sbert_comparison_summary.csv"))
    print("\n--- SBERT Model Comparison Summary ---")
    print(results_df)
    
    visualize_sbert_comparison(all_sbert_results, output_dir)
    print("\n===== SBERT Comparison Experiment Completed =====")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare performance of different SBERT models in the LSRM framework.")
    # Add all arguments
    parser.add_argument("--user_comments_csv", default="data/user_reviews_eng.csv")
    parser.add_argument("--user_content_col", default="comments_content")
    parser.add_argument("--user_true_label_col", default="index_num")
    parser.add_argument("--requirements_source_csv", default="data/product_functionalities.csv")
    parser.add_argument("--req_id_col", default="Id")
    parser.add_argument("--req_name_col", default="Functionality Name")
    parser.add_argument("--req_desc_col", default="Functionality Description")
    parser.add_argument("--req_exp_col", default="Functionality Experience")
    
    # Model and API arguments
    parser.add_argument("--llm_api_base", required=True)
    parser.add_argument("--llm_api_key", required=True)
    parser.add_argument("--llm_model_name", default="gemini-pro", help="LLM used for fixed keyword extraction.")
    parser.add_argument("--keyword_cache_file", default="cache/extracted_keywords.pkl", help="Path to cache extracted keywords.")
    parser.add_argument("--sbert_models", nargs='+', required=True, help="List of SBERT models to compare (paths or HF names).")
    
    # Experiment settings
    parser.add_argument("--threshold", type=float, default=0.55, help="Fixed similarity threshold for mapping.")
    parser.add_argument("--output_dir", default="results/sbert_comparison")
    parser.add_argument("--num_rows_comments", type=int, default=None)
    
    args = parser.parse_args()
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.dirname(args.keyword_cache_file)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    main(args)