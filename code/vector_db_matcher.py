import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from openpyxl import Workbook

import matching_app as m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query a local vector database for candidate-job matching.")
    parser.add_argument("--train", required=True, help="Path to Matching_training.xlsx")
    parser.add_argument("--validation", required=True, help="Path to Matching_validation.xlsx")
    parser.add_argument("--store-dir", default="vector_db_store", help="Directory where the vector store will be saved")
    parser.add_argument("--output", default="matching_validation_vector_db.xlsx", help="Output workbook path")
    parser.add_argument("--top-n", type=int, default=5, help="Top jobs to retrieve for each candidate")
    return parser.parse_args()


def build_dense_vectors(main_emb: np.ndarray, title_emb: np.ndarray, main_weight: float = 1.0, title_weight: float = 1.5) -> np.ndarray:
    dense = np.hstack([main_emb * main_weight, title_emb * title_weight]).astype(np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return dense / norms


def write_dataframe_sheet(workbook: Workbook, name: str, df: pd.DataFrame) -> None:
    ws = workbook.create_sheet(title=name)
    ws.append(list(df.columns))
    for row in df.itertuples(index=False, name=None):
        ws.append(list(row))


def build_vector_store(train_path: str, validation_path: str, store_dir: str) -> Dict[str, object]:
    training = m.load_training_dataset(train_path)
    validation = m.load_validation_dataset(validation_path)

    train_candidates = m.prepare_candidates(training.candidates)
    train_jobs = m.prepare_jobs(training.jobs)
    val_candidates = m.prepare_candidates(validation.candidates)
    val_jobs = m.prepare_jobs(validation.jobs)

    word_vectorizer, _, title_vectorizer = m.fit_vectorizers(
        all_candidate_docs=list(train_candidates["cv_clean"]) + list(val_candidates["cv_clean"]),
        all_job_docs=list(train_jobs["job_document"]) + list(val_jobs["job_document"]),
        all_titles=list(train_jobs["job_title_clean"]) + list(val_jobs["job_title_clean"]),
    )

    train_candidate_word = m.transform_texts(word_vectorizer, train_candidates["cv_clean"])
    train_candidate_title = m.transform_texts(title_vectorizer, train_candidates["cv_clean"])
    train_job_word = m.transform_texts(word_vectorizer, train_jobs["job_document"])
    train_job_title = m.transform_texts(title_vectorizer, train_jobs["job_title_clean"])

    svd_main, svd_title = m.fit_svd_embeddings(
        train_candidate_word=train_candidate_word,
        train_job_word=train_job_word,
        train_candidate_title=train_candidate_title,
        train_job_title=train_job_title,
    )

    val_candidate_word = m.transform_texts(word_vectorizer, val_candidates["cv_clean"])
    val_candidate_title = m.transform_texts(title_vectorizer, val_candidates["cv_clean"])
    val_job_word = m.transform_texts(word_vectorizer, val_jobs["job_document"])
    val_job_title = m.transform_texts(title_vectorizer, val_jobs["job_title_clean"])

    val_candidate_main_emb = m.transform_embeddings(svd_main, val_candidate_word)
    val_job_main_emb = m.transform_embeddings(svd_main, val_job_word)
    val_candidate_title_emb = m.transform_embeddings(svd_title, val_candidate_title)
    val_job_title_emb = m.transform_embeddings(svd_title, val_job_title)

    candidate_vectors = build_dense_vectors(val_candidate_main_emb, val_candidate_title_emb)
    job_vectors = build_dense_vectors(val_job_main_emb, val_job_title_emb)

    store_path = Path(store_dir)
    store_path.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        store_path / "vector_store.npz",
        candidate_vectors=candidate_vectors,
        job_vectors=job_vectors,
    )
    val_candidates[["CANDIDATE_ID", "CV_text", "cv_clean"]].to_csv(store_path / "candidates.csv", index=False)
    val_jobs[["Offer_ID", "Job_title_eng", "Job_post_eng", "job_document", "job_title_clean"]].to_csv(
        store_path / "jobs.csv", index=False
    )

    metadata = {
        "vector_space": "dense cosine similarity",
        "candidate_count": int(len(val_candidates)),
        "job_count": int(len(val_jobs)),
        "embedding_dimensions": int(candidate_vectors.shape[1]),
        "main_embedding_dims": int(val_candidate_main_emb.shape[1]),
        "title_embedding_dims": int(val_candidate_title_emb.shape[1]),
        "main_weight": 1.0,
        "title_weight": 1.5,
    }
    (store_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return {
        "validation_candidates": val_candidates,
        "validation_jobs": val_jobs,
        "candidate_vectors": candidate_vectors,
        "job_vectors": job_vectors,
        "metadata": metadata,
    }


def query_top_jobs(candidate_vectors: np.ndarray, job_vectors: np.ndarray, top_n: int) -> np.ndarray:
    similarity = candidate_vectors @ job_vectors.T
    top_indices = np.argsort(-similarity, axis=1)[:, :top_n]
    top_scores = np.take_along_axis(similarity, top_indices, axis=1)
    return top_indices, top_scores


def export_vector_results(
    candidates: pd.DataFrame,
    jobs: pd.DataFrame,
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    output_path: str,
) -> None:
    workbook = Workbook(write_only=True)

    candidate_text_map: Dict[int, str] = dict(zip(candidates["CANDIDATE_ID"].astype(int), candidates["CV_text"].astype(str)))
    job_ids = jobs["Offer_ID"].astype(str).to_numpy()
    job_titles = jobs["Job_title_eng"].astype(str).to_numpy()
    job_posts = jobs["Job_post_eng"].astype(str).to_numpy()

    top_rows: List[Dict[str, object]] = []
    candidate_ids = candidates["CANDIDATE_ID"].astype(int).to_numpy()
    for candidate_idx, candidate_id in enumerate(candidate_ids):
        for rank in range(top_indices.shape[1]):
            job_idx = int(top_indices[candidate_idx, rank])
            reason = m.summarize_overlap(
                candidate_text=candidate_text_map[int(candidate_id)],
                job_title=str(job_titles[job_idx]),
                job_text=str(job_posts[job_idx]),
            )
            top_rows.append(
                {
                    "CANDIDATE_ID": int(candidate_id),
                    "rank_for_candidate": rank + 1,
                    "Offer_ID": str(job_ids[job_idx]),
                    "vector_similarity_score": float(round(float(top_scores[candidate_idx, rank]) * 100.0, 6)),
                    "Job_title_eng": str(job_titles[job_idx]),
                    "match_reason": reason,
                }
            )

    top_df = pd.DataFrame(top_rows)
    write_dataframe_sheet(workbook, "VectorDB_Top_per_candidate", top_df)

    sorted_df = top_df.sort_values(
        by=["Job_title_eng", "vector_similarity_score", "CANDIDATE_ID"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    m.write_colored_dataframe_sheet(workbook, "VectorDB_sorted_by_job", sorted_df, group_column="Job_title_eng")

    info_df = pd.DataFrame(
        [
            {"item": "vector_space", "value": "dense cosine similarity"},
            {"item": "candidates", "value": len(candidates)},
            {"item": "jobs", "value": len(jobs)},
            {"item": "top_n", "value": top_indices.shape[1]},
            {"item": "output_type", "value": "vector database retrieval view"},
        ]
    )
    write_dataframe_sheet(workbook, "VectorDB_Info", info_df)
    workbook.save(output_path)


def main() -> int:
    args = parse_args()
    train_path = os.path.abspath(args.train)
    validation_path = os.path.abspath(args.validation)
    store_dir = os.path.abspath(args.store_dir)
    output_path = os.path.abspath(args.output)

    bundle = build_vector_store(train_path, validation_path, store_dir)
    top_indices, top_scores = query_top_jobs(
        candidate_vectors=bundle["candidate_vectors"],
        job_vectors=bundle["job_vectors"],
        top_n=max(1, int(args.top_n)),
    )
    export_vector_results(
        candidates=bundle["validation_candidates"],
        jobs=bundle["validation_jobs"],
        top_indices=top_indices,
        top_scores=top_scores,
        output_path=output_path,
    )

    print(f"Vector store saved to: {store_dir}")
    print(f"Vector workbook saved to: {output_path}")
    print(f"Candidates indexed: {bundle['metadata']['candidate_count']}")
    print(f"Jobs indexed: {bundle['metadata']['job_count']}")
    print(f"Embedding dimensions: {bundle['metadata']['embedding_dimensions']}")
    print(f"Top jobs per candidate: {top_indices.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
