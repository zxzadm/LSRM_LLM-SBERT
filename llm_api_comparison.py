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

# --- Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# --- Data Handling Functions ---
def read_and_split_sentences(csv_file, content_column='comments_content', nrows=None):
    df = pd.read_csv(csv_file, nrows=nrows) if nrows else pd.read_csv(csv_file)
    comments = df[content_column].fillna('').astype(str).tolist()
    sentences = [[s.strip() for s in re.split(r'[.!?\n]+', c) if s.strip()] for c in comments]
    return sentences, df

def load_product_functionalities(csv_file, name_col='Functionality Name', desc_col='Functionality Description', exp_col='Functionality Experience'):
    df = pd.read_csv(csv_file)
    names = df[name_col].astype(str).tolist()
    descriptions = df[desc_col].astype(str).fillna('').tolist()
    experiences = df[exp_col].astype(str).fillna('').tolist() if exp_col in df.columns else [''] * len(df)
    combined_descs = [f"{desc} {exp}".strip() for desc, exp in zip(descriptions, experiences)]
    return names, combined_descs

def load_true_labels(user_df, true_label_column, name_to_id_map):
    true_labels_sets = []
    for index_str in user_df[true_label_column].fillna('').astype(str):
        if index_str.strip():
            true_labels_sets.append({int(i.strip()) for i in index_str.split(',') if i.strip().isdigit()})
        else:
            true_labels_sets.append(set())
    return true_labels_sets

# --- Core Logic Functions ---
def extract_keywords_via_api(text, llm_client, model_name):
    # English prompt for keyword extraction
    system_prompt = "You are an expert in product requirement analysis for new energy vehicles. Your task is to accurately extract keywords reflecting specific functional requirements from user reviews."
    user_prompt = f"""Analyze the following user review sentence about a new energy vehicle: "{text}"
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

def generate_user_embeddings(sentences_per_comment, sbert_model, llm_client, llm_model_name):
    user_embeddings = []
    for sentences in tqdm(sentences_per_comment, desc=f"Generating user embeddings with {llm_model_name}"):
        comment_sentence_embs = []
        if not sentences:
            user_embeddings.append(np.array([]))
            continue
        original_sent_embs = sbert_model.encode(sentences, show_progress_bar=False)
        for i, sentence in enumerate(sentences):
            keywords = extract_keywords_via_api(sentence, llm_client, llm_model_name)
            keyword_emb = sbert_model.encode([' '.join(keywords)])[0]
            combined_emb = np.concatenate((original_sent_embs[i], keyword_emb))
            comment_sentence_embs.append(combined_emb)
        user_embeddings.append(np.array(comment_sentence_embs))
    return user_embeddings

def generate_feature_embeddings(func_names, func_descs, sbert_model):
    name_embs = sbert_model.encode(func_names, show_progress_bar=True)
    desc_embs = sbert_model.encode(func_descs, show_progress_bar=True)
    return [np.concatenate((n, d)) for n, d in zip(name_embs, desc_embs)]

def map_and_predict(user_embeddings, feature_embeddings, feature_names, threshold):
    predictions = []
    for comment_embs in user_embeddings:
        mapped_features = set()
        if comment_embs.size > 0:
            for sent_emb in comment_embs:
                similarities = util.cos_sim(sent_emb, feature_embeddings)[0]
                # Match all features above threshold for the sentence
                for i, score in enumerate(similarities):
                    if score >= threshold:
                        mapped_features.add(feature_names[i])
        predictions.append(", ".join(sorted(list(mapped_features))))
    return predictions

def evaluate(true_labels_sets, predicted_labels_sets, all_func_ids_set):
    precisions, recalls, f1s = [], [], []
    y_true_flat, y_pred_flat = [], []
    feature_list = sorted(list(all_func_ids_set))
    for true_set, pred_set in zip(true_labels_sets, predicted_labels_sets):
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
        y_true_flat.extend([1 if fid in true_set else 0 for fid in feature_list])
        y_pred_flat.extend([1 if fid in pred_set else 0 for fid in feature_list])
    return {"precision": np.mean(precisions), "recall": np.mean(recalls), "f1": np.mean(f1s), "accuracy": accuracy_score(y_true_flat, y_pred_flat)}

def main(args):
    output_dir = os.path.join(args.output_dir, args.llm_model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    user_sentences, user_df = read_and_split_sentences(args.user_comments_csv, args.user_content_col, args.num_rows_comments)
    func_names, func_descs, name_to_id_map, all_func_ids_set = load_product_functionalities(args.requirements_source_csv, args.req_name_col, args.req_desc_col, args.req_exp_col)
    
    # Initialize models
    sbert_model = SentenceTransformer(args.sbert_model_path)
    llm_client = OpenAI(base_url=args.llm_api_base, api_key=args.llm_api_key)

    # Generate embeddings
    user_embeddings = generate_user_embeddings(user_sentences, sbert_model, llm_client, args.llm_model_name)
    feature_embeddings = generate_feature_embeddings(func_names, func_descs, sbert_model)
    
    # Threshold analysis
    threshold_range = np.arange(0.35, 0.61, 0.01)
    results = []
    for threshold in tqdm(threshold_range, desc="Analyzing thresholds"):
        predicted_feature_names = map_and_predict(user_embeddings, feature_embeddings, func_names, threshold)
        true_labels = load_true_labels(user_df, args.user_true_label_col, name_to_id_map)
        predicted_labels = [{name_to_id_map[n] for n in p.split(', ') if n and n in name_to_id_map} for p in predicted_feature_names]
        metrics = evaluate(true_labels, predicted_labels, all_func_ids_set)
        results.append({"threshold": threshold, **metrics})
    
    # Save and visualize results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "threshold_performance.csv"), index=False)
    
    # Visualization logic here (as in previous scripts) ...
    plt.figure(figsize=(10, 6)); plt.plot(results_df["threshold"], results_df["f1"], marker='o'); plt.title(f'F1 Score vs. Threshold for {args.llm_model_name}'); plt.xlabel('Threshold'); plt.ylabel('F1 Score'); plt.grid(True); plt.savefig(os.path.join(output_dir, 'f1_vs_threshold.png')); plt.close()
    
    best_result = results_df.loc[results_df['f1'].idxmax()]
    print(f"\nBest performance for {args.llm_model_name}:\n{best_result}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare performance of different API-based LLMs in LSRM framework.")
    # Add all arguments as before...
    parser.add_argument("--user_comments_csv", default="data/user_reviews_eng.csv")
    parser.add_argument("--user_content_col", default="comments_content")
    parser.add_argument("--user_true_label_col", default="index_num")
    parser.add_argument("--requirements_source_csv", default="data/product_functionalities.csv")
    parser.add_argument("--req_name_col", default="Functionality Name")
    parser.add_argument("--req_desc_col", default="Functionality Description")
    parser.add_argument("--req_exp_col", default="Functionality Experience")
    parser.add_argument("--sbert_model_path", default="all-MiniLM-L6-v2")
    parser.add_argument("--llm_api_base", required=True)
    parser.add_argument("--llm_api_key", required=True)
    parser.add_argument("--llm_model_name", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--num_rows_comments", type=int, default=None)
    args = parser.parse_args()
    main(args)