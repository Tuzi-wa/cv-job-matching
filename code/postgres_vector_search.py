import argparse
import io
import os
import subprocess

import pandas as pd
from openpyxl import Workbook

import matching_app as m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query top candidate-job matches from PostgreSQL + pgvector.")
    parser.add_argument("--db-name", default="searchmatch_case2", help="PostgreSQL database name")
    parser.add_argument("--host", default=os.environ.get("PGHOST", "localhost"), help="PostgreSQL host")
    parser.add_argument("--port", default=os.environ.get("PGPORT", "5432"), help="PostgreSQL port")
    parser.add_argument("--user", default=os.environ.get("PGUSER"), help="PostgreSQL user")
    parser.add_argument("--top-n", type=int, default=5, help="Top jobs to retrieve for each candidate")
    parser.add_argument(
        "--output",
        default="Matching_validation_postgres_vector.xlsx",
        help="Workbook path for PostgreSQL vector-search results",
    )
    return parser.parse_args()


def build_psql_base_args(args: argparse.Namespace) -> list[str]:
    cmd = ["psql", "-v", "ON_ERROR_STOP=1", "-d", args.db_name]
    if args.host:
        cmd.extend(["-h", args.host])
    if args.port:
        cmd.extend(["-p", str(args.port)])
    if args.user:
        cmd.extend(["-U", args.user])
    return cmd


def run_psql_csv(args: argparse.Namespace, sql: str) -> pd.DataFrame:
    cmd = build_psql_base_args(args) + ["--csv", "-c", sql]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return pd.read_csv(io.StringIO(result.stdout))


def write_dataframe_sheet(workbook: Workbook, name: str, df: pd.DataFrame) -> None:
    ws = workbook.create_sheet(title=name)
    ws.append(list(df.columns))
    for row in df.itertuples(index=False, name=None):
        ws.append(list(row))


def build_result_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top_n = max(1, int(args.top_n))
    top_df = run_psql_csv(
        args,
        f"""
        with ranked as (
            select
                c.candidate_id,
                row_number() over (
                    partition by c.candidate_id
                    order by nearest.embedding <=> c.embedding
                ) as rank_for_candidate,
                nearest.offer_id,
                round(((1 - (nearest.embedding <=> c.embedding)) * 100)::numeric, 6) as vector_similarity_score,
                nearest.job_title_eng,
                c.cv_text,
                nearest.job_post_eng
            from candidate_vectors c
            join lateral (
                select offer_id, job_title_eng, job_post_eng, embedding
                from job_vectors
                order by embedding <=> c.embedding
                limit {top_n}
            ) nearest on true
        )
        select *
        from ranked
        order by candidate_id, rank_for_candidate;
        """,
    )
    if top_df.empty:
        return top_df, top_df, pd.DataFrame(columns=["item", "value"])

    top_df["match_reason"] = top_df.apply(
        lambda row: m.summarize_overlap(
            candidate_text=str(row["cv_text"]),
            job_title=str(row["job_title_eng"]),
            job_text=str(row["job_post_eng"]),
        ),
        axis=1,
    )
    top_df = top_df.rename(
        columns={
            "candidate_id": "CANDIDATE_ID",
            "offer_id": "Offer_ID",
            "job_title_eng": "Job_title_eng",
        }
    )
    top_df["vector_similarity_score"] = top_df["vector_similarity_score"].astype(float)
    top_df["CANDIDATE_ID"] = top_df["CANDIDATE_ID"].astype(int)
    top_df["Offer_ID"] = top_df["Offer_ID"].astype(str)
    top_df["CV_preview"] = top_df["cv_text"].astype(str).map(lambda text: text[:240] + ("..." if len(text) > 240 else ""))
    top_df["Job_requirements_preview"] = top_df["job_post_eng"].astype(str).map(
        lambda text: text[:240] + ("..." if len(text) > 240 else "")
    )
    sorted_df = top_df.sort_values(
        by=["Job_title_eng", "vector_similarity_score", "CANDIDATE_ID"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    info_df = run_psql_csv(args, "select item, value from vector_metadata order by item;")
    return top_df, sorted_df, info_df


def export_results(top_df: pd.DataFrame, sorted_df: pd.DataFrame, info_df: pd.DataFrame, output_path: str) -> None:
    workbook = Workbook(write_only=True)
    write_dataframe_sheet(workbook, "PGVector_TopMatches", top_df)
    if sorted_df.empty:
        write_dataframe_sheet(workbook, "PGVector_ByJob", sorted_df)
    else:
        m.write_colored_dataframe_sheet(workbook, "PGVector_ByJob", sorted_df, group_column="Job_title_eng")
    write_dataframe_sheet(workbook, "PGVector_Info", info_df)
    workbook.save(output_path)


def main() -> int:
    args = parse_args()
    output_path = os.path.abspath(args.output)
    top_df, sorted_df, info_df = build_result_frames(args)
    export_results(top_df, sorted_df, info_df, output_path)
    print(f"PostgreSQL vector workbook saved to: {output_path}")
    print(f"Rows exported: {len(top_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
