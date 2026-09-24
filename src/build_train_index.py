import joblib
import pandas as pd
import numpy as np
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "upwork-jobs.json"
INPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "upwork-jobs.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "vacancies_embeddings.joblib"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "vacancies_index.faiss"
INDEXED_DF_PATH = ARTIFACTS_DIR / "vacancies_indexed.csv"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_NAME = config.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = config.get("BATCH_SIZE", 128)

CATEGORIES_PATH = PROJECT_ROOT / config.get("CATEGORIES_FILE_PATH", "data/raw/categories.json")

with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
    categories = json.load(f)



df = pd.read_csv(INPUT_CSV_PATH)

def join_skills(x):
    if isinstance(x, list):
        return " ".join(x)
    if isinstance(x, str):
        return x.replace("[", "").replace("]", "").replace("'", "").replace(",", "")
    return ""


def create_embedding(corpus, model_name: str, batch_size: int, device: str, max_seq_length: int = 256):

    embedder = SentenceTransformer(model_name, device=device)
    embedder.max_seq_length = max_seq_length
    corpus_embeddings = embedder.encode_document(corpus,
                                                 batch_size=batch_size,
                                                 normalize_embeddings=True,
                                                 show_progress_bar=True,
                                                 convert_to_numpy=True,
                                                 convert_to_tensor=False)

    corpus_embeddings = np.ascontiguousarray(
        corpus_embeddings,
        dtype=np.float32
    )

    return corpus_embeddings


def create_index(df: pd.DataFrame, corpus_embeddings: np.ndarray, index_path: Path = FAISS_INDEX_PATH, csv_path: Path = INDEXED_DF_PATH):

    # if len(df) != corpus_embeddings.shape[0]:
    #     raise ValueError("Number of rows in df and corpus_embeddings must match.")

    if corpus_embeddings.ndim != 2:
        raise ValueError("corpus_embeddings must be a 2D numpy array.")

    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)

    # if index.ntotal != len(df):
    #     raise ValueError("Number of rows in df and index must match.")
    #
    # df["embedding_id"] = np.arange(len(df))
    if index_path is not None:
        faiss.write_index(index,
                      str(index_path))
    if csv_path is not None:
        df.to_csv(str(csv_path),
                    index=False)

    return index

def search_index(index, query: str, k=100):
    k = min(k, index.ntotal)
    query = f"search_query: {query}"
    query_embedding = create_embedding([query], model_name=MODEL_NAME, batch_size=BATCH_SIZE, device="cuda")
    scores, indices = index.search(query_embedding, k)
    return scores[0], indices[0]

def load_index(path_to_faiss_index: Path):
    index = faiss.read_index(str(path_to_faiss_index))
    return index

def assign_categories(path_to_csv: Path, categories: dict[str, str]):
    df = pd.read_csv(path_to_csv)

    # Support either singular or plural column name
    text_col = "text_for_embedding" if "text_for_embedding" in df.columns else "text_for_embeddings"
    if text_col not in df.columns:
        raise KeyError(f"Neither 'text_for_embedding' nor 'text_for_embeddings' found in {path_to_csv}. Available columns: {list(df.columns)}")

    category_keys = list(categories.keys())
    category_texts = [f"{k} - {v}" for k, v in categories.items()]

    # 1. Embed categories
    categories_embedding = create_embedding(
        category_texts,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        device="cuda",
    )
    categories_index = create_index(None, categories_embedding, None, None)

    # 2. Embed jobs (strip existing prefix if needed before querying)
    queries = [
        f"search_query: {text.replace('search_document: ', '')}"
        for text in df[text_col].astype(str)
    ]
    query_embeddings = create_embedding(
        queries, model_name=MODEL_NAME, batch_size=BATCH_SIZE, device="cuda"
    )

    # 3. Match and map categories
    _, indices = categories_index.search(query_embeddings.astype(np.float32), 1)
    matched_indices = indices.flatten()
    df["category"] = [category_keys[idx] for idx in matched_indices]

    df.to_csv(str(path_to_csv), index=False)

if __name__ == "__main__":

    # skills_text = df['skills'].apply(join_skills)
    # df["text_for_embedding"] = "search_document: Job: " + df['Job Title'].astype(
    #     str) + " | Skills: " + skills_text + " | Description: " + df[
    #                                'Description'].astype(str)
    # df["text_for_embedding"] = df["text_for_embedding"].fillna("").astype(str)
    # df = df[df["text_for_embedding"].str.strip() != ""].reset_index(drop=True)
    #
    # test_query = "3d modeling"
    # # corpus = df["text_for_embedding"].tolist()
    # # corpus_embeddings = create_embedding(corpus, MODEL_NAME, BATCH_SIZE, device="cuda")
    # # joblib.dump(corpus_embeddings, str(EMBEDDINGS_PATH))
    #
    # index = load_index(FAISS_INDEX_PATH)
    # scores, indices = search_index(index, test_query)
    #
    # indexed_csv = pd.read_csv(str(INDEXED_DF_PATH))
    #
    # results = indexed_csv["Job Title"].iloc[indices].copy()
    # results["semantic_score"] = scores
    #
    # print(f"""Top matching results: \n
    #           {"="*50} \n {results}""")

    assign_categories(INDEXED_DF_PATH, categories)







