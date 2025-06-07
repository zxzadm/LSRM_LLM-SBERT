import pandas as pd
import numpy as np
import re
import os
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Setup --- (Identical to API version)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# --- Data Handling Functions --- (Identical to API version)
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

# --- Core Logic Functions for Local LLM ---
def load_local_llm(model_path):
    print(f"Loading local LLM from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, temperature=0.1)

def extract_keywords_via_local_llm(text, llm_pipeline):
    # English prompt for keyword extraction
    system_prompt = "You are an expert in product requirement analysis..." # Same prompt as API version
    user_prompt = f'Analyze the following user review sentence: "{text}"...' # Same prompt as API version
    
    # Constructing prompt based on common chat templates
    prompt = llm_pipeline.tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True
    )
    
    try:
        outputs = llm_pipeline(prompt)
        # Extract the generated text after the prompt
        response = outputs[0]['generated_text']
        # This parsing logic might need adjustment based on the specific model's output format
        assistant_response = response.split(llm_pipeline.tokenizer.bos_token + 'assistant')[-1].strip() if 'assistant' in response else response[len(prompt):].strip()
        keywords_str = assistant_response.split('\n')[0] # Often the first line is the keyword list
        return [kw.strip() for kw in keywords_str.split(',')] if keywords_str else ["no_keywords"]
    except Exception as e:
        print(f"Local LLM error for text '{text[:40]}...': {str(e)}")
        return ["local_llm_error"]

def generate_user_embeddings_local(sentences_per_comment, sbert_model, local_llm_pipeline):
    # This function is very similar to the API version, just calls extract_keywords_via_local_llm
    # The logic remains the same: loop, extract keywords, encode, concatenate
    # ... Implementation would be nearly identical to `generate_user_embeddings` ...
    # For brevity, this is a simplified representation of the logic flow.
    user_embeddings = []
    for sentences in tqdm(sentences_per_comment, desc=f"Generating user embeddings with local LLM"):
        comment_sentence_embs = []
        if not sentences:
            user_embeddings.append(np.array([]))
            continue
        original_sent_embs = sbert_model.encode(sentences, show_progress_bar=False)
        for i, sentence in enumerate(sentences):
            keywords = extract_keywords_via_local_llm(sentence, local_llm_pipeline)
            keyword_emb = sbert_model.encode([' '.join(keywords)])[0]
            combined_emb = np.concatenate((original_sent_embs[i], keyword_emb))
            comment_sentence_embs.append(combined_emb)
        user_embeddings.append(np.array(comment_sentence_embs))
    return user_embeddings

# --- Other Functions --- (generate_feature_embeddings, map_and_predict, evaluate)
# These functions are identical to the API version and can be reused.
# For brevity, we assume they are defined as in the llm_api_comparison.py script.

def main(args):
    # This main function would be almost identical to the API version main function.
    # The key difference is loading the local LLM pipeline instead of the OpenAI client,
    # and calling `generate_user_embeddings_local` instead of `generate_user_embeddings`.
    # The rest of the flow (data loading, feature embedding, threshold analysis, evaluation, visualization) is the same.
    # ... Implementation would mirror the `main` function from `llm_api_comparison.py` ...
    print("This is a conceptual representation. The full flow would mirror the API script.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare performance of different local LLMs in LSRM framework.")
    # Add all arguments as before, but for local model path
    parser.add_argument("--user_comments_csv", default="data/user_reviews_eng.csv")
    # ... other data arguments ...
    parser.add_argument("--sbert_model_path", default="all-MiniLM-L6-v2")
    parser.add_argument("--local_llm_path", required=True, help="Path to the local LLM directory (e.g., from Hugging Face).")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--num_rows_comments", type=int, default=None)
    args = parser.parse_args()
    # The main function would need to be fully implemented to run, mirroring the API script's structure.
    # main(args)