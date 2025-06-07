import pandas as pd
import numpy as np
import re
import os
import argparse # For handling command-line arguments
from tqdm import tqdm # For displaying progress bars
from sklearn.model_selection import train_test_split # For splitting data

# --- Model-specific imports ---
import fasttext
import tensorflow_hub as hub
from gensim.models import KeyedVectors
from rank_bm25 import BM25Okapi
import jieba # Still needed for BM25 and Word2Vec if descriptions are Chinese, or replace with English tokenizer

# --- Evaluation-specific imports ---
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score # precision_score, recall_score, f1_score are calculated manually for multi-label
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Global settings ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# === 1. Data Loading and Preprocessing Functions ===
def read_and_split_sentences(csv_file_path, content_column='comments_content', nrows=None):
    """Reads a CSV file and splits comments into sentences based on punctuation."""
    print(f"Reading and splitting sentences from: {csv_file_path}")
    if nrows:
        df = pd.read_csv(csv_file_path, nrows=nrows)
    else:
        df = pd.read_csv(csv_file_path)
    
    # MODIFIED: Using the 'comments_content' column as per your new CSV.
    comments = df[content_column].fillna('').astype(str).tolist()
    # Using English and Chinese punctuation for splitting, assuming comments might still have mixed language or for robustness
    sentences = [re.split(r'[.!?。！？\n]+', comment) for comment in comments]
    sentences = [[sentence.strip() for sentence in sublist if sentence.strip() != ''] for sublist in sentences]
    print(f"Loaded {len(sentences)} comments.")
    return sentences

def load_requirement_descriptions(excel_file_path, sheet_name=0, 
                                  name_column='Functionality Name', 
                                  desc_column='Functionality Description',
                                  experience_column='Functionality Experience'): # Added experience column
    """Loads requirement descriptions from an Excel/CSV file."""
    print(f"Loading requirement descriptions from: {excel_file_path}")
    if excel_file_path.endswith('.xlsx'):
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    elif excel_file_path.endswith('.csv'):
        df = pd.read_csv(excel_file_path)
    else:
        raise ValueError("Unsupported file format for requirements. Please use .xlsx or .csv")

    # MODIFIED: Using new column names
    requirement_names = df[name_column].astype(str).tolist()
    requirement_descriptions_main = df[desc_column].astype(str).fillna('').tolist()
    
    # Optionally, combine with 'Functionality Experience'
    if experience_column in df.columns:
        requirement_experience = df[experience_column].astype(str).fillna('').tolist()
        requirement_descriptions_combined = [
            f"{main_desc} {exp_desc}".strip() 
            for main_desc, exp_desc in zip(requirement_descriptions_main, requirement_experience)
        ]
    else:
        requirement_descriptions_combined = requirement_descriptions_main

    print(f"Loaded {len(requirement_names)} requirement descriptions.")
    return requirement_names, requirement_descriptions_combined # Using combined descriptions

# === 2. Vectorization and Matching Functions for Baseline Models ===

# --- 2.1 FastText ---
def fasttext_sentence_to_vec(sentence, model):
    """Converts a sentence to its average FastText word vector."""
    # For English, simple space splitting might be okay, or use a proper English tokenizer.
    # If your FastText model is multilingual or Chinese, Jieba might still be relevant for mixed content.
    words = sentence.split() 
    word_vecs = [model.get_word_vector(word) for word in words]
    if not word_vecs:
        return np.zeros(model.get_dimension())
    return np.mean(word_vecs, axis=0)

def run_fasttext_experiment(user_comments, requirement_names, requirement_descriptions, model_path, threshold):
    model = fasttext.load_model(model_path)
    requirement_vecs = [fasttext_sentence_to_vec(desc, model) for desc in requirement_descriptions] 
    
    results = []
    for comment_idx, comment_sentences in enumerate(user_comments): 
        comment_features = set()
        for sentence in comment_sentences:
            sentence_vec = fasttext_sentence_to_vec(sentence, model)
            if np.all(sentence_vec == 0): continue
            
            similarities = cosine_similarity([sentence_vec], requirement_vecs)[0]
            if len(similarities) == 0: continue

            best_score_idx = np.argmax(similarities) # Taking only the top-1 match
            best_score = similarities[best_score_idx]
            
            if best_score > threshold:
                comment_features.add(requirement_names[best_score_idx])
        results.append({"Comment Index": comment_idx + 1, "Predicted Features": ", ".join(sorted(list(comment_features)))})
    return pd.DataFrame(results)

# --- 2.2 Universal Sentence Encoder (USE) ---
def use_sentence_to_vec(sentences_list, model):
    return model(sentences_list).numpy()

def run_use_experiment(user_comments, requirement_names, requirement_descriptions, model_path, threshold):
    model = hub.load(model_path) # Ensure this model_path points to an English or multilingual USE model
    requirement_vecs = use_sentence_to_vec(requirement_descriptions, model)
    
    results = []
    for comment_idx, comment_sentences in enumerate(user_comments):
        comment_features = set()
        if not comment_sentences: 
            results.append({"Comment Index": comment_idx + 1, "Predicted Features": ""})
            continue
            
        sentence_vecs_in_comment = use_sentence_to_vec(comment_sentences, model)
        
        for sentence_vec in sentence_vecs_in_comment:
            similarities = cosine_similarity([sentence_vec], requirement_vecs)[0]
            if len(similarities) == 0: continue

            best_score_idx = np.argmax(similarities)
            best_score = similarities[best_score_idx]
            
            if best_score > threshold:
                comment_features.add(requirement_names[best_score_idx])
        results.append({"Comment Index": comment_idx + 1, "Predicted Features": ", ".join(sorted(list(comment_features)))})
    return pd.DataFrame(results)

# --- 2.3 Word2Vec ---
def word2vec_sentence_to_vec(sentence, model):
    # MODIFIED: For English, use simple split. If Word2Vec model expects specific tokenization, adjust here.
    # If Word2Vec is Chinese or multilingual, Jieba might still be needed for some sentences.
    words = sentence.split() # Using simple space split for English
    # words = list(jieba.cut(sentence)) # Keep if Word2Vec model is Chinese
    word_vecs = [model[word] for word in words if word in model.key_to_index]
    if not word_vecs:
        return np.zeros(model.vector_size)
    return np.mean(word_vecs, axis=0)

def run_word2vec_experiment(user_comments, requirement_names, requirement_descriptions, model_path, threshold):
    # Ensure model_path points to an English or multilingual Word2Vec model
    model = KeyedVectors.load_word2vec_format(model_path, binary=False if model_path.endswith(('.txt', '.vec', '.model')) else True)
    requirement_vecs = [word2vec_sentence_to_vec(desc, model) for desc in requirement_descriptions]
    
    results = []
    for comment_idx, comment_sentences in enumerate(user_comments):
        comment_features = set()
        for sentence in comment_sentences:
            sentence_vec = word2vec_sentence_to_vec(sentence, model)
            if np.all(sentence_vec == 0): continue
            
            similarities = cosine_similarity([sentence_vec], requirement_vecs)[0]
            if len(similarities) == 0: continue

            best_score_idx = np.argmax(similarities)
            best_score = similarities[best_score_idx]
            
            if best_score > threshold:
                comment_features.add(requirement_names[best_score_idx])
        results.append({"Comment Index": comment_idx + 1, "Predicted Features": ", ".join(sorted(list(comment_features)))})
    return pd.DataFrame(results)

# --- 2.4 BM25 ---
def run_bm25_experiment(user_comments, requirement_names, requirement_descriptions, threshold):
    # MODIFIED: For English, use simple split. If descriptions are Chinese, Jieba is appropriate.
    tokenized_corpus = [desc.split() for desc in requirement_descriptions] # Simple space tokenization for English
    # tokenized_corpus = [list(jieba.cut(desc)) for desc in requirement_descriptions] # Keep if descriptions are Chinese
    bm25 = BM25Okapi(tokenized_corpus)
    
    results = []
    for comment_idx, comment_sentences in enumerate(user_comments):
        comment_features = set()
        for sentence in comment_sentences:
            tokenized_sentence = sentence.split() # Simple space tokenization for English
            # tokenized_sentence = list(jieba.cut(sentence)) # Keep if sentences are Chinese
            if not tokenized_sentence: continue
            
            scores = bm25.get_scores(tokenized_sentence)
            if len(scores) == 0: continue

            best_score_idx = np.argmax(scores)
            best_score = scores[best_score_idx]
            
            # BM25 threshold needs careful tuning, might not be in [0,1]
            if best_score > threshold: 
                comment_features.add(requirement_names[best_score_idx])
        results.append({"Comment Index": comment_idx + 1, "Predicted Features": ", ".join(sorted(list(comment_features)))})
    return pd.DataFrame(results)

# === 3. Performance Evaluation Function ===
def evaluate_performance(true_labels_list_of_sets, predicted_labels_list_of_sets, all_possible_feature_indices_set):
    # This function remains largely the same, as it works with sets of indices.
    sample_precisions = []
    sample_recalls = []
    sample_f1_scores = []
    
    feature_list_for_indexing = sorted(list(all_possible_feature_indices_set)) 

    all_true_binary_flat = []
    all_pred_binary_flat = []

    for true_set, pred_set in zip(true_labels_list_of_sets, predicted_labels_list_of_sets):
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        sample_precisions.append(precision)
        sample_recalls.append(recall)
        sample_f1_scores.append(f1)

        true_binary = [1 if i in true_set else 0 for i in feature_list_for_indexing]
        pred_binary = [1 if i in pred_set else 0 for i in feature_list_for_indexing]
        all_true_binary_flat.extend(true_binary)
        all_pred_binary_flat.extend(pred_binary)
        
    avg_precision = np.mean(sample_precisions) if sample_precisions else 0.0
    avg_recall = np.mean(sample_recalls) if sample_recalls else 0.0
    avg_f1 = np.mean(sample_f1_scores) if sample_f1_scores else 0.0
    
    accuracy = accuracy_score(all_true_binary_flat, all_pred_binary_flat) if all_true_binary_flat else 0.0

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "accuracy": accuracy
    }

def load_true_labels_and_map(user_comments_csv_path, 
                             product_functionality_csv_path, 
                             user_true_label_column='index_num', # Column in user CSV with comma-sep functionality indices
                             func_id_column='Id', # Column in product CSV with the ID/index
                             func_name_column='Functionality Name', # Column in product CSV with the name
                             nrows_user_comments=None):
    """Loads true labels from user comments CSV (as indices) and creates a name-to-index map from product functionality CSV."""
    print("Loading true labels and creating functionality map...")
    if nrows_user_comments:
        df_user_comments = pd.read_csv(user_comments_csv_path, nrows=nrows_user_comments)
    else:
        df_user_comments = pd.read_csv(user_comments_csv_path)
    
    # True labels are already indices in 'index_num' column
    true_labels_as_index_str_list = df_user_comments[user_true_label_column].fillna('').astype(str).tolist()
    true_labels_list_of_sets = []
    for index_str_entry in true_labels_as_index_str_list:
        current_true_set = set()
        if not index_str_entry.strip():
            true_labels_list_of_sets.append(current_true_set)
            continue
        indices = [int(idx.strip()) for idx in index_str_entry.split(',') if idx.strip().isdigit()]
        current_true_set.update(indices)
        true_labels_list_of_sets.append(current_true_set)

    # Load product functionalities to create map and get all possible indices
    if product_functionality_csv_path.endswith('.xlsx'):
        df_prod_func = pd.read_excel(product_functionality_csv_path)
    elif product_functionality_csv_path.endswith('.csv'):
        df_prod_func = pd.read_csv(product_functionality_csv_path)
    else:
        raise ValueError("Unsupported file format for product functionalities.")

    if func_id_column not in df_prod_func.columns or func_name_column not in df_prod_func.columns:
        raise ValueError(f"Required columns '{func_id_column}' or '{func_name_column}' not found in product functionality CSV.")

    # Create a mapping from functionality name to its ID/index
    # And a set of all possible functionality IDs/indices
    name_to_index_map = {}
    all_possible_feature_indices_set = set()
    for _, row in df_prod_func.iterrows():
        name = str(row[func_name_column])
        idx = int(row[func_id_column])
        name_to_index_map[name] = idx
        all_possible_feature_indices_set.add(idx)
    
    print(f"Loaded {len(true_labels_list_of_sets)} true label entries for user comments.")
    print(f"Created map for {len(name_to_index_map)} functionalities. Total unique functionality indices: {len(all_possible_feature_indices_set)}.")
    return true_labels_list_of_sets, name_to_index_map, all_possible_feature_indices_set

def convert_predicted_features_to_indices(predicted_df, name_to_index_map, feature_column='Predicted Features'): # Changed column name
    """Converts predicted requirement names to sets of indices."""
    predicted_labels_list_of_sets = []
    if feature_column not in predicted_df.columns:
        print(f"Error: Column '{feature_column}' not found in predicted_df. Available columns: {predicted_df.columns}")
        # Fallback: if 'Features' exists (from original code), use it.
        if 'Features' in predicted_df.columns:
            feature_column = 'Features'
            print(f"Using fallback column: 'Features'")
        else:
            return [set() for _ in range(len(predicted_df))] 

    for _, row in predicted_df.iterrows():
        current_pred_set = set()
        # Handle potential float (NaN) or other non-string types
        feature_val = row[feature_column]
        if pd.isna(feature_val) or not str(feature_val).strip():
            predicted_labels_list_of_sets.append(current_pred_set)
            continue

        pred_names = [name.strip() for name in str(feature_val).split(',') if name.strip()]
        for name in pred_names:
            if name in name_to_index_map:
                current_pred_set.add(name_to_index_map[name])
            # else: # Optional: print warning for names not in map
            #     print(f"Warning: Predicted feature name '{name}' not found in functionality map.")
        predicted_labels_list_of_sets.append(current_pred_set)
    return predicted_labels_list_of_sets

# === 4. Main Execution and Command-line Argument Handling ===
def main():
    parser = argparse.ArgumentParser(description="Run baseline models for requirement mapping with validation/test split.")
    parser.add_argument("model_type", choices=["fasttext", "use", "word2vec", "bm25"], help="Type of baseline model to run.")
    
    # MODIFIED: Updated default CSV names and added specific column names for clarity
    parser.add_argument("--user_comments_csv", default="user_comments_en.csv", # Example new name
                        help="Path to the user comments CSV file.")
    parser.add_argument("--user_content_col", default="comments_content", 
                        help="Name of the column in user_comments_csv with comment text.")
    parser.add_argument("--user_true_label_col", default="index_num", 
                        help="Name of the column in user_comments_csv with true label indices (comma-separated).")
    
    parser.add_argument("--requirements_source_csv", default="product_functionalities_en.csv", # Example new name
                        help="Path to the CSV file containing all requirement names, IDs, and descriptions.")
    parser.add_argument("--req_id_col", default="Id", 
                        help="Name of the column in requirements_source_csv with the functionality ID/index.")
    parser.add_argument("--req_name_col", default="Functionality Name", 
                        help="Name of the column in requirements_source_csv with the functionality name.")
    parser.add_argument("--req_desc_col", default="Functionality Description",
                        help="Name of the column in requirements_source_csv with the main functionality description.")
    parser.add_argument("--req_exp_col", default="Functionality Experience",
                        help="Name of the column in requirements_source_csv with the experience description (optional).")

    # Model paths (ensure these are suitable for English or multilingual)
    parser.add_argument("--fasttext_model_path", default="crawl-300d-2M-subword.bin", help="Path to FastText model (e.g., English model).") # Example English model
    parser.add_argument("--use_model_path", default="https://tfhub.dev/google/universal-sentence-encoder-large/5", help="Path or URL to USE model (e.g., large English model).") # Example large English model
    parser.add_argument("--word2vec_model_path", default="GoogleNews-vectors-negative300.bin", help="Path to Word2Vec model (e.g., Google News).") # Example English model
    
    parser.add_argument("--output_dir", default="baseline_results_english_optimized", help="Directory to save results.")
    parser.add_argument("--test_size", type=float, default=0.3, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for data splitting.")
    parser.add_argument("--num_rows_comments", type=int, default=None, help="Number of rows to read from comments_csv for quick testing.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. Load all data
    all_user_comments_sentences = read_and_split_sentences(args.user_comments_csv, 
                                                           content_column=args.user_content_col, 
                                                           nrows=args.num_rows_comments)
    all_req_names, all_req_descs = load_requirement_descriptions(args.requirements_source_csv,
                                                                 name_column=args.req_name_col,
                                                                 desc_column=args.req_desc_col,
                                                                 experience_column=args.req_exp_col)
    
    all_true_labels_sets, name_to_idx_map, all_indices_set = load_true_labels_and_map(
        args.user_comments_csv, 
        args.requirements_source_csv,
        user_true_label_column=args.user_true_label_col,
        func_id_column=args.req_id_col,
        func_name_column=args.req_name_col,
        nrows_user_comments=args.num_rows_comments
    )
    
    min_len = min(len(all_user_comments_sentences), len(all_true_labels_sets))
    all_user_comments_sentences = all_user_comments_sentences[:min_len]
    all_true_labels_sets = all_true_labels_sets[:min_len]
    
    indices = list(range(len(all_user_comments_sentences)))
    # Stratified split might be better if label distribution is very imbalanced, but simple split for now.
    train_val_indices, test_indices = train_test_split(indices, test_size=args.test_size, random_state=args.random_state)
    
    val_user_comments = [all_user_comments_sentences[i] for i in train_val_indices]
    val_true_labels_sets = [all_true_labels_sets[i] for i in train_val_indices]
    
    test_user_comments = [all_user_comments_sentences[i] for i in test_indices]
    test_true_labels_sets = [all_true_labels_sets[i] for i in test_indices]

    print(f"Validation set size: {len(val_user_comments)} comments.")
    print(f"Test set size: {len(test_user_comments)} comments.")

    threshold_ranges = { # These ranges might need adjustment for English models / new data
        "fasttext": np.arange(0.3, 0.81, 0.05),
        "use": np.arange(0.3, 0.81, 0.05), 
        "word2vec": np.arange(0.3, 0.81, 0.05),
        "bm25": np.arange(0, 25, 1) # BM25 scores are not normalized, range can vary significantly
    }
    current_threshold_range = threshold_ranges.get(args.model_type, np.arange(0.4, 0.71, 0.05))
    
    best_f1_val = -1
    optimal_threshold = current_threshold_range[0] # Default to first if no better found
    val_threshold_results = []

    print(f"\n--- Optimizing threshold for {args.model_type.upper()} on Validation Set ---")
    for threshold_val in tqdm(current_threshold_range, desc=f"Validating {args.model_type}"):
        predicted_df_val = None
        if args.model_type == "fasttext":
            predicted_df_val = run_fasttext_experiment(val_user_comments, all_req_names, all_req_descs, args.fasttext_model_path, threshold_val)
        elif args.model_type == "use":
            # USE model path might be a local directory or a TF Hub URL
            predicted_df_val = run_use_experiment(val_user_comments, all_req_names, all_req_descs, args.use_model_path, threshold_val)
        elif args.model_type == "word2vec":
            predicted_df_val = run_word2vec_experiment(val_user_comments, all_req_names, all_req_descs, args.word2vec_model_path, threshold_val)
        elif args.model_type == "bm25":
            predicted_df_val = run_bm25_experiment(val_user_comments, all_req_names, all_req_descs, threshold_val)
        
        if predicted_df_val is not None:
            # MODIFIED: Use the new column name for predicted features
            predicted_indices_val = convert_predicted_features_to_indices(predicted_df_val, name_to_idx_map, feature_column='Predicted Features')
            metrics_val = evaluate_performance(val_true_labels_sets, predicted_indices_val, all_indices_set)
            val_threshold_results.append({"threshold": threshold_val, **metrics_val})
            
            if metrics_val['f1'] > best_f1_val:
                best_f1_val = metrics_val['f1']
                optimal_threshold = threshold_val
    
    print(f"Optimal threshold found for {args.model_type.upper()} on validation set: {optimal_threshold:.3f} (F1: {best_f1_val:.4f})")

    val_threshold_df = pd.DataFrame(val_threshold_results)
    val_threshold_path = os.path.join(args.output_dir, f"{args.model_type}_validation_threshold_performance.csv")
    val_threshold_df.to_csv(val_threshold_path, index=False)
    print(f"Saved validation threshold performance to {val_threshold_path}")

    if not val_threshold_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(val_threshold_df["threshold"], val_threshold_df["precision"], label='Precision (Val)', marker='o')
        plt.plot(val_threshold_df["threshold"], val_threshold_df["recall"], label='Recall (Val)', marker='s')
        plt.plot(val_threshold_df["threshold"], val_threshold_df["f1"], label='F1 Score (Val)', marker='^')
        plt.plot(val_threshold_df["threshold"], val_threshold_df["accuracy"], label='Accuracy (Val)', marker='d')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Validation Performance vs Threshold for {args.model_type.upper()}')
        plt.legend()
        plt.grid(True)
        val_plot_path = os.path.join(args.output_dir, f"{args.model_type}_validation_performance_plot.png")
        plt.savefig(val_plot_path)
        print(f"Saved validation performance plot to {val_plot_path}")
        plt.close()
    else:
        print("No validation results to plot.")


    print(f"\n--- Evaluating {args.model_type.upper()} on Test Set with Optimal Threshold: {optimal_threshold:.3f} ---")
    predicted_df_test = None
    if args.model_type == "fasttext":
        predicted_df_test = run_fasttext_experiment(test_user_comments, all_req_names, all_req_descs, args.fasttext_model_path, optimal_threshold)
    elif args.model_type == "use":
        predicted_df_test = run_use_experiment(test_user_comments, all_req_names, all_req_descs, args.use_model_path, optimal_threshold)
    elif args.model_type == "word2vec":
        predicted_df_test = run_word2vec_experiment(test_user_comments, all_req_names, all_req_descs, args.word2vec_model_path, optimal_threshold)
    elif args.model_type == "bm25":
        predicted_df_test = run_bm25_experiment(test_user_comments, all_req_names, all_req_descs, optimal_threshold)

    final_metrics_test = {}
    if predicted_df_test is not None:
        # MODIFIED: Use the new column name for predicted features
        predicted_indices_test = convert_predicted_features_to_indices(predicted_df_test, name_to_idx_map, feature_column='Predicted Features')
        final_metrics_test = evaluate_performance(test_true_labels_sets, predicted_indices_test, all_indices_set)
        
        test_pred_path = os.path.join(args.output_dir, f"{args.model_type}_test_predictions_optimal_thresh.csv")
        predicted_df_test.to_csv(test_pred_path, index=False)
        print(f"Saved test predictions to {test_pred_path}")

    print(f"\n--- Final Performance for {args.model_type.upper()} on Test Set ---")
    print(f"Optimal Threshold (from validation): {optimal_threshold:.3f}")
    for metric_name, metric_value in final_metrics_test.items():
        print(f"{metric_name.capitalize()}: {metric_value:.4f}")

    final_metrics_summary = {
        "model_type": args.model_type,
        "optimal_threshold_on_validation": optimal_threshold,
        "validation_f1_at_optimal_threshold": best_f1_val,
        **{f"test_{k}": v for k,v in final_metrics_test.items()}
    }
    summary_df = pd.DataFrame([final_metrics_summary])
    summary_path = os.path.join(args.output_dir, f"{args.model_type}_final_performance_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved final performance summary to {summary_path}")

if __name__ == "__main__":
    main()