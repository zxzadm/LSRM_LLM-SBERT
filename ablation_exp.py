import pandas as pd
import numpy as np
import re
import os
import argparse # For command-line arguments (optional, but good practice)
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI # Or your preferred LLM client
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split # For splitting data
import matplotlib as mpl
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
# import jieba.analyse # For Chinese TF-IDF, replace with English equivalent if needed
# For English TF-IDF, sklearn's TfidfVectorizer is often used directly with English text.
# If 'jieba.analyse' was specifically for TF-IDF keyword extraction,
# we might need an English alternative or adjust its usage.
# For now, assuming TF-IDF will work on English text after splitting.
import seaborn as sns

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 设置matplotlib参数
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# --- 1. Data Loading and Preprocessing ---
def read_and_split_sentences_eng(csv_file_path, content_column='comments_content', nrows=None):
    """Reads CSV with English comments and splits them into sentences."""
    print(f"Reading and splitting sentences from: {csv_file_path}")
    if nrows:
        df = pd.read_csv(csv_file_path, nrows=nrows)
    else:
        df = pd.read_csv(csv_file_path)
    
    comments = df[content_column].fillna('').astype(str).tolist()
    # Using English punctuation for splitting
    sentences_per_comment = [re.split(r'[.!?\n]+', comment) for comment in comments]
    sentences_per_comment = [[s.strip() for s in sublist if s.strip()] for sublist in sentences_per_comment]
    print(f"Loaded {len(df)} comments, split into sentences.")
    return sentences_per_comment, df # Return df for labels

def load_product_functionalities_eng(csv_file_path, 
                                     id_col='Id',
                                     name_col='Functionality Name', 
                                     desc_col='Functionality Description',
                                     exp_col='Functionality Experience'):
    """Loads product functionalities from CSV."""
    print(f"Loading product functionalities from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    ids = df[id_col].tolist()
    names = df[name_col].astype(str).tolist()
    descriptions_main = df[desc_col].astype(str).fillna('').tolist()
    
    if exp_col in df.columns:
        descriptions_exp = df[exp_col].astype(str).fillna('').tolist()
        combined_descriptions = [f"{main} {exp}".strip() for main, exp in zip(descriptions_main, descriptions_exp)]
    else:
        combined_descriptions = descriptions_main
        
    name_to_id_map = {name: fid for name, fid in zip(names, ids)}
    all_func_ids_set = set(ids)

    print(f"Loaded {len(names)} product functionalities.")
    return names, combined_descriptions, name_to_id_map, all_func_ids_set

def load_true_labels_from_df(user_df, true_label_column='index_num'):
    """Extracts true label sets (of indices) from the user DataFrame."""
    true_labels_as_index_str_list = user_df[true_label_column].fillna('').astype(str).tolist()
    true_labels_list_of_sets = []
    for index_str_entry in true_labels_as_index_str_list:
        current_true_set = set()
        if index_str_entry.strip():
            indices = [int(idx.strip()) for idx in index_str_entry.split(',') if idx.strip().isdigit()]
            current_true_set.update(indices)
        true_labels_list_of_sets.append(current_true_set)
    return true_labels_list_of_sets

# --- 2. Embedding and Keyword Extraction ---
def generate_sbert_embeddings(sentences_per_comment, sbert_model):
    """Generates SBERT embeddings for lists of sentences."""
    all_embeddings = []
    print("Generating SBERT sentence embeddings:")
    for sentences_list in tqdm(sentences_per_comment, desc="Processing comments for SBERT"):
        if sentences_list:
            all_embeddings.append(sbert_model.encode(sentences_list, show_progress_bar=False))
        else:
            all_embeddings.append(np.array([])) # Handle comments with no sentences after splitting
    return all_embeddings

def extract_keywords_llm_eng(text, client, model_name="gemini-pro", use_cot=True): # Changed default model
    """Extract keywords using LLM API with English prompt (CoT optional)."""
    # MODIFIED: Prompts are now in English
    if use_cot:
        system_prompt = "You are an expert focused on product requirement analysis for new energy vehicles. Your task is to accurately extract keywords reflecting specific functional requirements from user reviews."
        user_prompt = f"""Analyze the following user review about a new energy vehicle: "{text}"
Please follow this analytical framework:
1. Identify specific car functions, performance aspects, or user experience points mentioned in the review.
2. Convert these points into concise functional requirement keywords.
3. Extract only genuine requirements explicitly expressed or implied in the review, ignoring irrelevant content.
Requirement keywords should be:
- Specific and precise (e.g., "automatic parking" is better than "parking").
- Expressing function or performance (e.g., "fast charging technology", "heated seats").
- Using standard automotive industry terminology where applicable.
Example:
Review: "The seats get cold in winter, I wish it had a heating function. Also, the navigation is often inaccurate."
Keywords: seat heating, navigation accuracy
Please output a comma-separated list of keywords directly, with no other explanation:"""
    else: # Simple prompt without CoT
        system_prompt = "Your task is to extract keywords from user reviews."
        user_prompt = f"""Extract core functional requirement keywords from the following new energy vehicle user review: "{text}"
Output a comma-separated list of keywords directly:"""
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        completion = client.chat.completions.create(
            model=model_name, # Ensure this model is suitable for English
            messages=messages,
            temperature=0.2 # Lower temperature for more deterministic keyword extraction
        )
        keywords_str = completion.choices[0].message.content
        if not keywords_str or keywords_str.strip().lower() == "none": # Handle empty or "none" responses
            return ["no_keywords_extracted"] # Return a placeholder
            
        # Splitting by comma, then stripping whitespace from each keyword
        return [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
    
    except Exception as e:
        print(f"LLM API call error for text '{text[:50]}...': {str(e)}. Returning placeholder.")
        return ["llm_api_error"] # Return a placeholder

def extract_keywords_tfidf_eng(text, top_n=5, stop_words='english'):
    """Extract keywords using TF-IDF for English text."""
    if not text or len(text.split()) < 3: # Check word count for very short texts
        return ["no_keywords_extracted"]
    
    try:
        # Using sklearn's TfidfVectorizer for English
        vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000) # Limit features for single doc
        # Fit and transform the single document (text)
        # TfidfVectorizer expects an iterable of documents, so pass [text]
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get scores for the document
        if tfidf_matrix.nnz == 0: # No terms found after stopword removal/vectorization
             return ["no_keywords_extracted"]

        doc_vector = tfidf_matrix[0]
        # Get indices of top_n features
        # For a single document, we can sum scores if needed or just take highest tf-idf terms.
        # A simpler way for a single doc is to look at the terms with highest tf-idf scores.
        # This part needs adjustment as TF-IDF is typically for a corpus.
        # For single document keyword extraction, simple term frequency or other methods might be more direct.
        # However, to mimic 'jieba.analyse.extract_tags', we can try this:
        scores = doc_vector.data
        indices = doc_vector.indices
        
        if len(scores) == 0:
            return ["no_keywords_extracted"]

        top_indices = indices[np.argsort(scores)[-top_n:]] # Get indices of top N scores
        keywords = [feature_names[i] for i in top_indices]
        
        return keywords if keywords else ["no_keywords_extracted"]
    except Exception as e:
        print(f"TF-IDF extraction error: {str(e)}")
        return ["tfidf_extraction_error"]

def generate_user_embeddings_for_ablation(
    sentences_per_comment, 
    sbert_embeddings_per_comment, # Pre-computed SBERT embeddings for original sentences
    sbert_model, # SBERT model instance for encoding keywords
    llm_client, 
    ablation_type, 
    llm_model_name="gemini-pro" # Ensure this is an English-capable model
):
    """Generates user requirement embeddings based on ablation settings."""
    final_user_embeddings_per_comment = []
    
    print(f"\nGenerating user embeddings for ablation type: {ablation_type} using LLM: {llm_model_name}")
    
    for i, comment_sents in enumerate(tqdm(sentences_per_comment, desc=f"Processing comments for {ablation_type}")):
        original_sbert_embs = sbert_embeddings_per_comment[i]
        
        if not comment_sents: # If a comment has no sentences
            final_user_embeddings_per_comment.append(np.array([]))
            continue

        comment_final_sentence_embeddings = []
        for sent_idx, sentence_text in enumerate(comment_sents):
            original_sent_emb = original_sbert_embs[sent_idx] if original_sbert_embs.ndim > 1 else original_sbert_embs # Handle single sentence case
            
            if ablation_type == "full_model":
                keywords = extract_keywords_llm_eng(sentence_text, llm_client, llm_model_name, use_cot=True)
                kw_emb = sbert_model.encode([' '.join(keywords)])[0] if keywords else np.zeros_like(original_sent_emb)
                final_emb = np.concatenate((original_sent_emb, kw_emb))
            elif ablation_type == "wo_llm":
                final_emb = original_sent_emb # Only SBERT sentence embedding
            elif ablation_type == "wo_sbert":
                keywords = extract_keywords_llm_eng(sentence_text, llm_client, llm_model_name, use_cot=True)
                final_emb = sbert_model.encode([' '.join(keywords)])[0] if keywords else np.zeros(sbert_model.get_sentence_embedding_dimension()) # Use a zero vector of SBERT's dim
            elif ablation_type == "simple_ke": # Using TF-IDF
                keywords = extract_keywords_tfidf_eng(sentence_text)
                kw_emb = sbert_model.encode([' '.join(keywords)])[0] if keywords else np.zeros_like(original_sent_emb)
                final_emb = np.concatenate((original_sent_emb, kw_emb))
            elif ablation_type == "wo_cot":
                keywords = extract_keywords_llm_eng(sentence_text, llm_client, llm_model_name, use_cot=False)
                kw_emb = sbert_model.encode([' '.join(keywords)])[0] if keywords else np.zeros_like(original_sent_emb)
                final_emb = np.concatenate((original_sent_emb, kw_emb))
            else:
                raise ValueError(f"Unknown ablation type: {ablation_type}")
            comment_final_sentence_embeddings.append(final_emb)
        final_user_embeddings_per_comment.append(np.array(comment_final_sentence_embeddings)) # Store as numpy array
        
    return final_user_embeddings_per_comment

def calculate_product_feature_embeddings_for_ablation(
    func_names, 
    func_descriptions, 
    sbert_model, 
    ablation_type
):
    """Calculates product feature embeddings based on ablation settings."""
    print(f"\nCalculating product feature embeddings for ablation type: {ablation_type}")
    
    # Base SBERT embeddings for names and descriptions
    # Ensure func_names and func_descriptions are not empty
    name_embs = sbert_model.encode(func_names if func_names else [''])
    desc_embs = sbert_model.encode(func_descriptions if func_descriptions else [''])

    if ablation_type in ["full_model", "simple_ke", "wo_cot"]:
        # Concatenate name and description embeddings
        # Check if either list was empty during encoding
        if not func_names or not func_descriptions: # Should not happen if data is loaded correctly
             return np.array([])
        combined_embs = [np.concatenate((n_emb, d_emb)) for n_emb, d_emb in zip(name_embs, desc_embs)]
    elif ablation_type == "wo_llm": # Corresponds to User side only having SBERT sentence
        # For product side, let's use description only to match SBERT sentence directly
        combined_embs = desc_embs
    elif ablation_type == "wo_sbert": # Corresponds to User side only having LLM keyword SBERT embedding
        # For product side, let's use name only (as a proxy for "keyword-like" representation)
        combined_embs = name_embs
    else:
        raise ValueError(f"Unknown ablation type for feature embeddings: {ablation_type}")
        
    return np.array(combined_embs) # Return as numpy array

# --- 3. Comparison and Evaluation (largely similar to baseline, but adapted) ---
def compare_and_map_embeddings(user_embeddings_per_comment, 
                               product_feature_embeddings, 
                               product_functionality_names, 
                               threshold):
    """Compares user comment embeddings to product feature embeddings."""
    mapped_results = []
    for comment_idx, single_comment_sentence_embeddings in enumerate(user_embeddings_per_comment):
        matched_functionalities_for_comment = set()
        if single_comment_sentence_embeddings.ndim == 0 or single_comment_sentence_embeddings.size == 0: # Handle empty arrays
            mapped_results.append({"Comment Index": comment_idx, "Predicted Functionalities": ""})
            continue
        
        # Ensure it's a 2D array if multiple sentences, or reshape if single sentence
        embs_to_iterate = single_comment_sentence_embeddings
        if embs_to_iterate.ndim == 1: # Single sentence in the comment
            embs_to_iterate = embs_to_iterate.reshape(1, -1)

        for sentence_embedding in embs_to_iterate:
            if sentence_embedding.size == 0: continue # Skip if somehow an empty sentence emb made it
            similarities = util.cos_sim(sentence_embedding, product_feature_embeddings)[0]
            # Match all functionalities above threshold for this sentence
            for func_idx, sim_score in enumerate(similarities):
                if sim_score >= threshold:
                    matched_functionalities_for_comment.add(product_functionality_names[func_idx])
                    
        mapped_results.append({
            "Comment Index": comment_idx, 
            "Predicted Functionalities": ", ".join(sorted(list(matched_functionalities_for_comment)))
        })
    return pd.DataFrame(mapped_results)

def evaluate_predictions(true_labels_sets, predicted_df, name_to_id_map, all_func_ids_set):
    """Evaluates predictions against true labels."""
    predicted_functionalities_str_list = predicted_df["Predicted Functionalities"].fillna('').astype(str).tolist()
    
    predicted_labels_sets = []
    for pred_str in predicted_functionalities_str_list:
        current_pred_ids = set()
        if pred_str:
            names = [name.strip() for name in pred_str.split(',') if name.strip()]
            for name in names:
                if name in name_to_id_map:
                    current_pred_ids.add(name_to_id_map[name])
        predicted_labels_sets.append(current_pred_ids)

    # Reusing the core logic from baseline's evaluate_performance
    # (Assuming evaluate_performance from baseline script is available or reimplemented here)
    # For simplicity, let's quickly reimplement the core part:
    sample_precisions, sample_recalls, sample_f1s = [], [], []
    y_true_flat, y_pred_flat = [], []
    
    # Sort all_func_ids_set to have a consistent order for binary vectors
    feature_list_for_indexing = sorted(list(all_func_ids_set))


    for true_set, pred_set in zip(true_labels_sets, predicted_labels_sets):
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        sample_precisions.append(prec)
        sample_recalls.append(rec)
        sample_f1s.append(f1)

        true_binary = [1 if fid in true_set else 0 for fid in feature_list_for_indexing]
        pred_binary = [1 if fid in pred_set else 0 for fid in feature_list_for_indexing]
        y_true_flat.extend(true_binary)
        y_pred_flat.extend(pred_binary)

    return {
        "precision": np.mean(sample_precisions) if sample_precisions else 0.0,
        "recall": np.mean(sample_recalls) if sample_recalls else 0.0,
        "f1": np.mean(sample_f1s) if sample_f1s else 0.0,
        "accuracy": accuracy_score(y_true_flat, y_pred_flat) if y_true_flat else 0.0
    }

# --- 4. Visualization (can reuse or adapt baseline's) ---
def visualize_ablation_validation_performance(val_results_all_ablations, output_dir):
    """Visualizes F1 score vs. threshold for each ablation type on validation set."""
    plt.figure(figsize=(12, 8), dpi=300)
    for ablation_type, results_df in val_results_all_ablations.items():
        if not results_df.empty:
            plt.plot(results_df["threshold"], results_df["f1"], label=f'{ablation_type.replace("_", " ").upper()} F1 (Val)', marker='o', markersize=4, linestyle='--')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1 Score (Validation)', fontsize=12)
    plt.title('Validation F1 Score vs. Threshold for Ablation Variants', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.5)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ablation_validation_f1_vs_threshold.png")
    plt.savefig(plot_path)
    print(f"Saved ablation validation performance plot to {plot_path}")
    plt.close()

def visualize_final_ablation_comparison(final_test_results, output_dir):
    """Visualizes final test performance of ablation variants (bar chart)."""
    ablation_types = list(final_test_results.keys())
    metrics = ['precision', 'recall', 'f1', 'accuracy']
    
    plot_data = {metric: [final_test_results[ab_type].get(metric, 0) for ab_type in ablation_types] for metric in metrics}
    df_plot = pd.DataFrame(plot_data, index=[ab_type.replace("_", " ").upper() for ab_type in ablation_types])
    
    df_plot.plot(kind='bar', figsize=(15, 8), colormap='viridis')
    plt.title('Final Test Performance of Ablation Variants', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Ablation Variant', fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title='Metrics')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_ablation_test_comparison.png'))
    print(f"Saved final ablation test comparison plot to {os.path.join(output_dir, 'final_ablation_test_comparison.png')}")
    plt.close()


# --- 5. Main Ablation Study Orchestration ---
def run_ablation_study_optimized(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # --- Load Data ---
    all_user_comments_sents, user_df = read_and_split_sentences_eng(
        args.user_comments_csv, 
        content_column=args.user_content_col, 
        nrows=args.num_rows_comments
    )
    all_func_names, all_func_descs, name_to_id_map, all_func_ids_set = load_product_functionalities_eng(
        args.requirements_source_csv,
        id_col=args.req_id_col,
        name_col=args.req_name_col,
        desc_col=args.req_desc_col,
        exp_col=args.req_exp_col
    )
    all_true_labels_sets = load_true_labels_from_df(user_df, args.user_true_label_col)

    # --- Split Data ---
    # Ensure all lists are of the same length before splitting
    min_len = min(len(all_user_comments_sents), len(all_true_labels_sets))
    all_user_comments_sents = all_user_comments_sents[:min_len]
    all_true_labels_sets = all_true_labels_sets[:min_len]
    
    indices = list(range(len(all_user_comments_sents)))
    train_val_indices, test_indices = train_test_split(indices, test_size=args.test_size, random_state=args.random_state)
    
    val_user_sents = [all_user_comments_sents[i] for i in train_val_indices]
    val_true_labels = [all_true_labels_sets[i] for i in train_val_indices]
    test_user_sents = [all_user_comments_sents[i] for i in test_indices]
    test_true_labels = [all_true_labels_sets[i] for i in test_indices]

    print(f"Validation set size: {len(val_user_sents)} comments.")
    print(f"Test set size: {len(test_user_sents)} comments.")

    # --- Initialize Models ---
    sbert_model = SentenceTransformer(args.sbert_model_path) # Ensure this is an English-compatible SBERT
    llm_client = OpenAI(base_url=args.llm_api_base, api_key=args.llm_api_key) # Configure your LLM client

    # Pre-generate SBERT embeddings for original sentences (validation and test)
    # This is done once to save computation during ablation and threshold search
    val_sbert_embeddings_original = generate_sbert_embeddings(val_user_sents, sbert_model)
    test_sbert_embeddings_original = generate_sbert_embeddings(test_user_sents, sbert_model)


    # --- Ablation Setup ---
    ablation_variants = ["full_model", "wo_llm", "wo_sbert", "simple_ke", "wo_cot"]
    threshold_range_for_ablation = np.arange(0.40, 0.61, 0.01) # As requested: 0.4 to 0.6, step 0.01
    
    final_test_results_all_ablations = {}
    validation_perf_all_ablations = {} # To store validation curves data

    for ablation_type in ablation_variants:
        print(f"\n===== PROCESSING ABLATION VARIANT: {ablation_type.upper()} =====")
        
        # --- Generate Product Feature Embeddings for this ablation type (once) ---
        product_feature_embs_ablation = calculate_product_feature_embeddings_for_ablation(
            all_func_names, all_func_descs, sbert_model, ablation_type
        )
        if product_feature_embs_ablation.size == 0:
            print(f"Skipping {ablation_type} due to empty product feature embeddings.")
            continue

        # --- Generate User Embeddings for Validation Set for this ablation type (once) ---
        val_user_embs_ablation = generate_user_embeddings_for_ablation(
            val_user_sents, val_sbert_embeddings_original, sbert_model, llm_client, 
            ablation_type, args.llm_model_name
        )

        # --- Optimize Threshold on Validation Set ---
        best_f1_val = -1
        optimal_threshold_val = threshold_range_for_ablation[0] # Default
        current_val_threshold_results_list = []

        print(f"\n--- Optimizing threshold for {ablation_type.upper()} on Validation Set ---")
        for th_val in tqdm(threshold_range_for_ablation, desc=f"Validating {ablation_type}"):
            predicted_df_val = compare_and_map_embeddings(
                val_user_embs_ablation, product_feature_embs_ablation, all_func_names, th_val
            )
            metrics_val = evaluate_predictions(val_true_labels, predicted_df_val, name_to_id_map, all_func_ids_set)
            current_val_threshold_results_list.append({"threshold": th_val, **metrics_val})

            if metrics_val['f1'] > best_f1_val:
                best_f1_val = metrics_val['f1']
                optimal_threshold_val = th_val
        
        val_threshold_df = pd.DataFrame(current_val_threshold_results_list)
        validation_perf_all_ablations[ablation_type] = val_threshold_df # Store for plotting
        val_threshold_path = os.path.join(args.output_dir, f"{ablation_type}_validation_threshold_perf.csv")
        val_threshold_df.to_csv(val_threshold_path, index=False)
        print(f"Optimal threshold for {ablation_type}: {optimal_threshold_val:.2f} (Val F1: {best_f1_val:.4f})")
        print(f"Saved {ablation_type} validation threshold performance to {val_threshold_path}")


        # --- Evaluate on Test Set with Optimal Threshold ---
        print(f"\n--- Evaluating {ablation_type.upper()} on Test Set (Optimal Th: {optimal_threshold_val:.2f}) ---")
        # Generate User Embeddings for Test Set for this ablation type
        test_user_embs_ablation = generate_user_embeddings_for_ablation(
            test_user_sents, test_sbert_embeddings_original, sbert_model, llm_client, 
            ablation_type, args.llm_model_name
        )
        
        predicted_df_test = compare_and_map_embeddings(
            test_user_embs_ablation, product_feature_embs_ablation, all_func_names, optimal_threshold_val
        )
        final_metrics_test = evaluate_predictions(test_true_labels, predicted_df_test, name_to_id_map, all_func_ids_set)
        final_test_results_all_ablations[ablation_type] = final_metrics_test
        
        test_pred_path = os.path.join(args.output_dir, f"{ablation_type}_test_predictions_optimal_thresh.csv")
        predicted_df_test.to_csv(test_pred_path, index=False)
        print(f"Saved {ablation_type} test predictions to {test_pred_path}")
        print(f"Test Performance for {ablation_type}: F1={final_metrics_test['f1']:.4f}, P={final_metrics_test['precision']:.4f}, R={final_metrics_test['recall']:.4f}, A={final_metrics_test['accuracy']:.4f}")

    # --- Save and Visualize Overall Results ---
    summary_data = []
    for ab_type, metrics in final_test_results_all_ablations.items():
        # Find optimal threshold and val F1 for this ablation type again (already computed but for clarity)
        opt_th = validation_perf_all_ablations[ab_type].iloc[validation_perf_all_ablations[ab_type]['f1'].idxmax()]['threshold']
        val_f1 = validation_perf_all_ablations[ab_type]['f1'].max()
        summary_data.append({
            "ablation_variant": ab_type,
            "optimal_threshold_on_validation": opt_th,
            "validation_f1_at_optimal_threshold": val_f1,
            **{f"test_{k}": v for k, v in metrics.items()}
        })
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(args.output_dir, "ablation_study_final_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved final ablation study summary to {summary_path}")

    visualize_ablation_validation_performance(validation_perf_all_ablations, args.output_dir)
    visualize_final_ablation_comparison(final_test_results_all_ablations, args.output_dir)

    print("\n===== Ablation Study Completed =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ablation Study for LSRM with threshold optimization.")
    
    # Data paths - using similar names as baseline for consistency
    parser.add_argument("--user_comments_csv", default="user_reviews_eng.csv", help="Path to user comments CSV.")
    parser.add_argument("--user_content_col", default="comments_content", help="Column for comment text.")
    parser.add_argument("--user_true_label_col", default="index_num", help="Column for true label indices.")
    
    parser.add_argument("--requirements_source_csv", default="product_functionalities.csv", help="Path to product functionalities CSV.")
    parser.add_argument("--req_id_col", default="Id", help="Column for functionality ID.")
    parser.add_argument("--req_name_col", default="Functionality Name", help="Column for functionality name.")
    parser.add_argument("--req_desc_col", default="Functionality Description", help="Column for main functionality description.")
    parser.add_argument("--req_exp_col", default="Functionality Experience", help="Column for experience description.")

    # Model paths and configs
    parser.add_argument("--sbert_model_path", default="all-MiniLM-L6-v2", # Example English SBERT
                        help="Path or HuggingFace name of the SBERT model.")
    parser.add_argument("--llm_api_base", default="YOUR_LLM_API_BASE", help="Base URL for LLM API.")
    parser.add_argument("--llm_api_key", default="YOUR_LLM_API_KEY", help="API Key for LLM.")
    parser.add_argument("--llm_model_name", default="gemini-pro", # Or gpt-3.5-turbo, etc.
                        help="Name of the LLM model to use for keyword extraction.")
    
    # Experiment settings
    parser.add_argument("--output_dir", default="lsrm_ablation_study_optimized_eng", help="Directory for results.")
    parser.add_argument("--test_size", type=float, default=0.3, help="Test set proportion.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for splitting.")
    parser.add_argument("--num_rows_comments", type=int, default=None, help="Max rows from comments CSV for quick testing.")
    
    args = parser.parse_args()
    
    # !!! IMPORTANT: Replace with your actual API details or use environment variables !!!
    if args.llm_api_base == "YOUR_LLM_API_BASE" or args.llm_api_key == "YOUR_LLM_API_KEY":
        print("!!! ERROR: Please provide your LLM API base URL and key via arguments  !!!")
        print("!!! or modify the script with your actual credentials.                 !!!")
        exit(1)
        
    run_ablation_study_optimized(args)