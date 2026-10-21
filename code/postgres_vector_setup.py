import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import vector_db_matcher as vdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a PostgreSQL + pgvector store for candidate/job embeddings.")
    parser.add_argument("--train", required=True, help="Path to Matching_training.xlsx")
    parser.add_argument("--validation", required=True, help="Path to Matching_validation.xlsx")
    parser.add_argument("--db-name", default="searchmatch_case2", help="PostgreSQL database name")
    parser.add_argument("--host", default=os.environ.get("PGHOST", "localhost"), help="PostgreSQL host")
    parser.add_argument("--port", default=os.environ.get("PGPORT", "5432"), help="PostgreSQL port")
    parser.add_argument("--user", default=os.environ.get("PGUSER"), help="PostgreSQL user")
    parser.add_argument(
        "--store-dir",
        default="vector_db_store_pg",
        help="Directory for a local copy of the generated vectors and metadata",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the candidate/job vector tables before loading data",
    )
    return parser.parse_args()


def build_psql_base_args(args: argparse.Namespace, db_name: str) -> list[str]:
    cmd = ["psql", "-v", "ON_ERROR_STOP=1", "-d", db_name]
    if args.host:
        cmd.extend(["-h", args.host])
    if args.port:
        cmd.extend(["-p", str(args.port)])
    if args.user:
        cmd.extend(["-U", args.user])
    return cmd


def run_psql_sql(args: argparse.Namespace, db_name: str, sql: str, capture_output: bool = True) -> subprocess.CompletedProcess:
    cmd = build_psql_base_args(args, db_name) + ["-Atqc", sql]
    return subprocess.run(cmd, check=True, text=True, capture_output=capture_output)


def run_psql_command(args: argparse.Namespace, db_name: str, command: str) -> None:
    cmd = build_psql_base_args(args, db_name) + ["-c", command]
    subprocess.run(cmd, check=True, text=True)


def ensure_database(args: argparse.Namespace) -> None:
    db_name = args.db_name.replace("'", "''")
    result = run_psql_sql(args, "postgres", f"select 1 from pg_database where datname = '{db_name}';")
    if result.stdout.strip() == "1":
        return

    createdb_cmd = ["createdb", args.db_name]
    if args.host:
        createdb_cmd.extend(["-h", args.host])
    if args.port:
        createdb_cmd.extend(["-p", str(args.port)])
    if args.user:
        createdb_cmd.extend(["-U", args.user])
    subprocess.run(createdb_cmd, check=True, text=True)


def ensure_pgvector(args: argparse.Namespace) -> None:
    result = run_psql_sql(args, args.db_name, "select name from pg_available_extensions where name = 'vector';")
    if result.stdout.strip() != "vector":
        raise RuntimeError(
            "pgvector extension is not available in this PostgreSQL installation. "
            "Install pgvector first, then rerun this script."
        )

    try:
        run_psql_command(args, args.db_name, "create extension if not exists vector;")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to enable pgvector. Check that the extension is installed and that your PostgreSQL user "
            "has permission to run CREATE EXTENSION vector."
        ) from exc


def vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vector.tolist()) + "]"


def prepare_export_frames(bundle: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    candidates = bundle["validation_candidates"].copy()
    jobs = bundle["validation_jobs"].copy()
    candidate_vectors = bundle["candidate_vectors"]
    job_vectors = bundle["job_vectors"]
    metadata = bundle["metadata"]

    candidate_export = candidates.loc[:, ["CANDIDATE_ID", "CV_text", "cv_clean"]].copy()
    candidate_export["embedding"] = [vector_literal(vec) for vec in candidate_vectors]
    candidate_export = candidate_export.rename(
        columns={"CANDIDATE_ID": "candidate_id", "CV_text": "cv_text", "cv_clean": "cv_clean"}
    )

    job_export = jobs.loc[:, ["Offer_ID", "Job_title_eng", "Job_post_eng", "job_document", "job_title_clean"]].copy()
    job_export["embedding"] = [vector_literal(vec) for vec in job_vectors]
    job_export = job_export.rename(
        columns={
            "Offer_ID": "offer_id",
            "Job_title_eng": "job_title_eng",
            "Job_post_eng": "job_post_eng",
            "job_document": "job_document",
            "job_title_clean": "job_title_clean",
        }
    )
    return candidate_export, job_export, metadata


def create_schema(args: argparse.Namespace, embedding_dims: int, reset: bool) -> None:
    if reset:
        run_psql_command(
            args,
            args.db_name,
            """
            drop table if exists vector_metadata;
            drop table if exists candidate_vectors;
            drop table if exists job_vectors;
            """,
        )

    schema_sql = f"""
    create table if not exists candidate_vectors (
        candidate_id bigint primary key,
        cv_text text,
        cv_clean text,
        embedding vector({embedding_dims}) not null
    );

    create table if not exists job_vectors (
        offer_id text primary key,
        job_title_eng text,
        job_post_eng text,
        job_document text,
        job_title_clean text,
        embedding vector({embedding_dims}) not null
    );

    create table if not exists vector_metadata (
        item text primary key,
        value text not null
    );

    truncate table vector_metadata;
    truncate table candidate_vectors;
    truncate table job_vectors;
    """
    run_psql_command(args, args.db_name, schema_sql)


def copy_dataframe(args: argparse.Namespace, db_name: str, df: pd.DataFrame, table_name: str, columns: list[str]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        csv_path = Path(tmp.name)
        df.to_csv(tmp, index=False)

    try:
        quoted_columns = ", ".join(columns)
        escaped_path = str(csv_path).replace("'", "''")
        command = f"\\copy {table_name} ({quoted_columns}) from '{escaped_path}' with (format csv, header true)"
        run_psql_command(args, db_name, command)
    finally:
        csv_path.unlink(missing_ok=True)


def insert_metadata(args: argparse.Namespace, metadata: dict[str, object]) -> None:
    rows = [
        ("vector_space", "pgvector cosine distance"),
        ("candidate_count", str(metadata["candidate_count"])),
        ("job_count", str(metadata["job_count"])),
        ("embedding_dimensions", str(metadata["embedding_dimensions"])),
        ("main_embedding_dims", str(metadata["main_embedding_dims"])),
        ("title_embedding_dims", str(metadata["title_embedding_dims"])),
        ("main_weight", str(metadata["main_weight"])),
        ("title_weight", str(metadata["title_weight"])),
    ]
    value_chunks = []
    for item, value in rows:
        safe_item = item.replace("'", "''")
        safe_value = value.replace("'", "''")
        value_chunks.append(f"('{safe_item}', '{safe_value}')")
    values = ", ".join(value_chunks)
    run_psql_command(
        args,
        args.db_name,
        f"insert into vector_metadata (item, value) values {values} on conflict (item) do update set value = excluded.value;",
    )


def create_indexes(args: argparse.Namespace) -> None:
    run_psql_command(
        args,
        args.db_name,
        """
        create index if not exists candidate_vectors_embedding_idx
        on candidate_vectors using ivfflat (embedding vector_cosine_ops) with (lists = 20);

        create index if not exists job_vectors_embedding_idx
        on job_vectors using ivfflat (embedding vector_cosine_ops) with (lists = 50);

        analyze candidate_vectors;
        analyze job_vectors;
        """,
    )


def main() -> int:
    args = parse_args()
    ensure_database(args)
    ensure_pgvector(args)

    bundle = vdb.build_vector_store(
        train_path=os.path.abspath(args.train),
        validation_path=os.path.abspath(args.validation),
        store_dir=os.path.abspath(args.store_dir),
    )

    candidate_export, job_export, metadata = prepare_export_frames(bundle)
    create_schema(args, embedding_dims=int(metadata["embedding_dimensions"]), reset=args.reset)
    copy_dataframe(args, args.db_name, candidate_export, "candidate_vectors", ["candidate_id", "cv_text", "cv_clean", "embedding"])
    copy_dataframe(
        args,
        args.db_name,
        job_export,
        "job_vectors",
        ["offer_id", "job_title_eng", "job_post_eng", "job_document", "job_title_clean", "embedding"],
    )
    insert_metadata(args, metadata)
    create_indexes(args)

    print(f"Database ready: {args.db_name}")
    print(f"Candidates loaded: {len(candidate_export)}")
    print(f"Jobs loaded: {len(job_export)}")
    print(f"Embedding dimensions: {metadata['embedding_dimensions']}")
    print(f"Local vector snapshot saved to: {os.path.abspath(args.store_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
