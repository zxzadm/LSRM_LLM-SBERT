import pandas as pd
import numpy as np
import re
import os
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from openai import OpenAI # Using OpenAI client for LLM access
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import accuracy_score # For evaluation

os.environ["TOKENIZERS_PARALLELISM"] = "false"
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# --- 1. Data Loading and Preprocessing ---
def read_and_split_sentences_eng(csv_file_path, content_column='comments_content', nrows=None):
    print(f"Reading and splitting sentences from: {csv_file_path}")
    df = pd.read_csv(csv_file_path, nrows=nrows) if nrows else pd.read_csv(csv_file_path)
    comments = df[content_column].fillna('').astype(str).tolist()
    sentences_per_comment = [[s.strip() for s in re.split(r'[.!?\n]+', c) if s.strip()] for c in comments]
    print(f"Loaded {len(df)} comments.")
    return sentences_per_comment, df

def load_product_functionalities_eng(csv_file_path, id_col='Id', name_col='Functionality Name'):
    print(f"Loading product functionalities from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    ids = df[id_col].tolist()
    names = df[name_col].astype(str).tolist()
    name_to_id_map = {name: fid for name, fid in zip(names, ids)}
    all_func_ids_set = set(ids)
    print(f"Loaded {len(names)} product functionalities.")
    return names, name_to_id_map, all_func_ids_set

def load_true_labels_from_df(user_df, true_label_column='index_num'):
    true_labels_list_of_sets = []
    for index_str_entry in user_df[true_label_column].fillna('').astype(str):
        current_true_set = set()
        if index_str_entry.strip():
            current_true_set.update(int(idx.strip()) for idx in index_str_entry.split(',') if idx.strip().isdigit())
        true_labels_list_of_sets.append(current_true_set)
    return true_labels_list_of_sets

# --- 2. QA-based Zero-Shot "Mapping" (More aligned with paper's QA Answer Extraction) ---

QUESTION_TEMPLATES_FOR_FUNCTIONALITY = [
    "Is the following review sentence discussing or mentioning '{functionality}'?",
    "Does the review sentence provide any information about '{functionality}'?",
    "Based on the sentence, is '{functionality}' a topic of discussion?",
    "What, if anything, does the sentence say about '{functionality}'?" # More open-ended
]

def ask_llm_as_qa(context_sentence, functionality_name, llm_client, llm_model_name):
    """
    Prompts an LLM to act as a Question-Answering model to determine if the context_sentence
    is relevant to the functionality_name.
    Returns:
        - True if the LLM's response indicates relevance.
        - False if the LLM's response indicates no relevance or uncertainty.
        - None if there's an API error.
    """
    # Pick a question template
    question_template = np.random.choice(QUESTION_TEMPLATES_FOR_FUNCTIONALITY)
    question = question_template.format(functionality=functionality_name)

    # This prompt is crucial for guiding the LLM.
    # We are asking for a Yes/No style answer primarily, or an extraction.
    system_prompt = (
        "You are an AI assistant performing an extractive question-answering task. "
        "Based ONLY on the provided 'Review Sentence', answer the 'Question'. "
        "If the sentence explicitly or strongly implicitly discusses the functionality mentioned in the question, "
        "try to extract the relevant part of the sentence or confirm its relevance. "
        "If the sentence is NOT relevant to the functionality, or does not contain information to answer, "
        "respond with 'No, not relevant' or 'Not mentioned'."
    )
    user_prompt = f"Review Sentence: \"{context_sentence}\"\n\nQuestion: \"{question}\"\n\nAnswer (Is it relevant? If yes, what part? If no, state 'No, not relevant'):"

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        completion = llm_client.chat.completions.create(
            model=llm_model_name,
            messages=messages,
            temperature=0.0, # Very low temperature for factual, less creative answers
            max_tokens=60
        )
        answer = completion.choices[0].message.content.strip().lower()

        # Analyze the answer to determine relevance
        # These are heuristics and can be improved.
        negative_responses = ["not relevant", "not mentioned", "no information", "cannot determine", "doesn't say", "no."]
        if any(neg_res in answer for neg_res in negative_responses):
            return False
        
        # If the answer is very short and just "yes", it might be a weak signal.
        # A more robust check would be if the answer contains parts of the context or the functionality name again.
        # For simplicity now, any answer that is not explicitly negative is treated as potentially relevant.
        # The original paper's QA models (T5, RoBERTa on SQuAD) would extract a span.
        # If an LLM is used as a proxy, we check if it *didn't* say no.
        if len(answer) > 0: # It provided some answer other than a clear negative
             # Further check: if the answer contains the functionality name (could be a good sign)
            if functionality_name.lower() in answer:
                return True
            # If the answer is short and generic like "yes" but doesn't elaborate, it's ambiguous.
            # For this baseline, let's be a bit lenient if it's not a clear "no".
            # However, the paper's fine-tuned QA models are more precise.
            if "yes" in answer.split() and len(answer.split()) <= 2 : # e.g. "Yes." or "Yes, it is."
                # This might be too lenient, depends on LLM's verbosity
                pass # Let it pass for now, could be refined

            # If it extracts something from the sentence, it's likely relevant
            # This is hard to check without comparing to original sentence perfectly
            return True # Default to true if not explicitly negative.
        
        return False # Default if answer is empty or very ambiguous after checks

    except Exception as e:
        print(f"LLM-QA API error for context '{context_sentence[:30]}...': {str(e)}")
        return None # Indicates an error

def run_llm_qa_baseline_experiment(
    user_comments_sentences_list,
    all_functionality_names,
    llm_client,
    llm_model_name
):
    print(f"\nRunning LLM-as-QA Zero-Shot Baseline using LLM: {llm_model_name}")
    results = []
    
    for comment_idx, sentences_in_comment in enumerate(tqdm(user_comments_sentences_list, desc="QA Baseline: Processing Comments")):
        comment_mapped_functionalities = set()
        if not sentences_in_comment:
            results.append({"Comment Index": comment_idx + 1, "Predicted Functionalities": ""})
            continue
            
        for sentence_text in sentences_in_comment:
            if not sentence_text: continue
            for func_name in all_functionality_names:
                is_relevant = ask_llm_as_qa(sentence_text, func_name, llm_client, llm_model_name)
                if is_relevant is True: # Explicitly check for True
                    comment_mapped_functionalities.add(func_name)
                # We could also handle 'is_relevant is None' (API error) if needed, e.g., by logging
                    
        results.append({
            "Comment Index": comment_idx + 1, 
            "Predicted Functionalities": ", ".join(sorted(list(comment_mapped_functionalities)))
        })
    return pd.DataFrame(results)

# --- 3. Evaluation (Same as before) ---
def evaluate_predictions(true_labels_sets, predicted_df, name_to_id_map, all_func_ids_set):
    # This function can be reused from your previous ablation script or baseline script.
    # Ensure it correctly converts predicted names to IDs and calculates P, R, F1, Acc.
    # For brevity, assuming this function is correctly implemented as in your `ablation_exp.py`
    # or the baseline script I provided earlier.
    # Key: it takes true_labels (sets of IDs), predicted_df (with "Predicted Functionalities" as names),
    # name_to_id_map, and all_func_ids_set.

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

    sample_precisions, sample_recalls, sample_f1s = [], [], []
    y_true_flat, y_pred_flat = [], []
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


# --- 4. Main Execution ---
def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_user_comments_sents, user_df = read_and_split_sentences_eng(
        args.user_comments_csv, content_column=args.user_content_col, nrows=args.num_rows_comments
    )
    all_func_names, name_to_id_map, all_func_ids_set = load_product_functionalities_eng(
        args.requirements_source_csv, id_col=args.req_id_col, name_col=args.req_name_col
    )
    all_true_labels_sets = load_true_labels_from_df(user_df, args.user_true_label_col)

    min_len = min(len(all_user_comments_sents), len(all_true_labels_sets))
    all_user_comments_sents = all_user_comments_sents[:min_len]
    all_true_labels_sets = all_true_labels_sets[:min_len]
    
    # For this zero-shot baseline, we typically evaluate on a test set directly.
    # No validation set needed for threshold tuning with this QA approach.
    # We'll use the standard train/test split from args to define the test set.
    indices = list(range(len(all_user_comments_sents)))
    if args.test_size < 1.0: # If test_size is a fraction, perform split
        _, test_indices = train_test_split(indices, test_size=args.test_size, random_state=args.random_state)
        test_user_sents = [all_user_comments_sents[i] for i in test_indices]
        test_true_labels = [all_true_labels_sets[i] for i in test_indices]
    else: # Use all data as test data if test_size is 1.0 or more
        print("Using all loaded data as test data for zero-shot evaluation.")
        test_user_sents = all_user_comments_sents
        test_true_labels = all_true_labels_sets


    print(f"Test set size for QA baseline: {len(test_user_sents)} comments.")

    llm_as_qa_client = OpenAI(base_url=args.llm_api_base, api_key=args.llm_api_key)

    predicted_df_test = run_llm_qa_baseline_experiment(
        test_user_sents,
        all_func_names,
        llm_as_qa_client,
        args.llm_model_name
    )
    
    final_metrics_test = evaluate_predictions(test_true_labels, predicted_df_test, name_to_id_map, all_func_ids_set)
    
    test_pred_path = os.path.join(args.output_dir, f"llm_as_qa_baseline_test_predictions_{args.llm_model_name}.csv")
    predicted_df_test.to_csv(test_pred_path, index=False)
    print(f"Saved LLM-as-QA baseline test predictions to {test_pred_path}")

    print(f"\n--- Final Performance for LLM-as-QA Baseline ({args.llm_model_name}) on Test Set ---")
    for metric_name, metric_value in final_metrics_test.items():
        print(f"{metric_name.capitalize()}: {metric_value:.4f}")

    summary_data = {
        "baseline_type": "LLM_as_QA_ZeroShot",
        "llm_model_used": args.llm_model_name,
        **{f"test_{k}": v for k, v in final_metrics_test.items()}
    }
    summary_df = pd.DataFrame([summary_data])
    summary_path = os.path.join(args.output_dir, f"llm_as_qa_baseline_summary_{args.llm_model_name}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved LLM-as-QA baseline final performance summary to {summary_path}")

    print("\n===== LLM-as-QA Baseline Run Completed =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-as-QA Zero-Shot Baseline for user req to product func mapping.")
    
    parser.add_argument("--user_comments_csv", default="user_reviews_eng.csv")
    parser.add_argument("--user_content_col", default="comments_content")
    parser.add_argument("--user_true_label_col", default="index_num")
    
    parser.add_argument("--requirements_source_csv", default="product_functionalities.csv")
    parser.add_argument("--req_id_col", default="Id")
    parser.add_argument("--req_name_col", default="Functionality Name")

    parser.add_argument("--llm_api_base", required=True, help="Base URL for the LLM API.")
    parser.add_argument("--llm_api_key", required=True, help="API Key for the LLM.")
    parser.add_argument("--llm_model_name", default="gemini-pro", help="LLM model to use as QA engine.")
    
    parser.add_argument("--output_dir", default="results_llm_as_qa_baseline_eng")
    parser.add_argument("--test_size", type=float, default=0.3, help="Proportion for test split (if <1.0), or use all data if >=1.0.")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--num_rows_comments", type=int, default=None)
    
    args = parser.parse_args()
        
    main(args)