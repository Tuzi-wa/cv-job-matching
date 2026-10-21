from pathlib import Path
import html
import re

import pandas as pd
import streamlit as st


DEFAULT_BASE = Path(__file__).resolve().parent.parent
DEFAULT_MATCHING_FILE = DEFAULT_BASE / "results" / "Matching_validation_scored.xlsx"
DEFAULT_VECTOR_FILE = DEFAULT_BASE / "results" / "Matching_validation_vector_db.xlsx"
DEFAULT_PG_VECTOR_FILE = DEFAULT_BASE / "results" / "Matching_validation_postgres_vector.xlsx"


st.set_page_config(page_title="CV job matching", layout="wide")


@st.cache_data(show_spinner=False)
def load_matching_data(matching_path: str) -> dict:
    path = Path(matching_path)
    return {
        "top5": pd.read_excel(path, sheet_name="Top_5_per_candidate"),
        "top20": pd.read_excel(path, sheet_name="Top_20_per_candidate"),
        "job_sorted": pd.read_excel(path, sheet_name="Top_5_sorted_by_job_title"),
        "candidates": pd.read_excel(path, sheet_name="Candidates"),
        "jobs": pd.read_excel(path, sheet_name="Job_posts"),
    }


@st.cache_data(show_spinner=False)
def load_vector_data(vector_path: str) -> dict:
    path = Path(vector_path)
    return {
        "top": pd.read_excel(path, sheet_name="VectorDB_Top_per_candidate"),
        "job_sorted": pd.read_excel(path, sheet_name="VectorDB_sorted_by_job"),
        "info": pd.read_excel(path, sheet_name="VectorDB_Info"),
    }


@st.cache_data(show_spinner=False)
def load_pg_vector_data(pg_vector_path: str) -> dict:
    path = Path(pg_vector_path)
    return {
        "top": pd.read_excel(path, sheet_name="PGVector_TopMatches"),
        "job_sorted": pd.read_excel(path, sheet_name="PGVector_ByJob"),
        "info": pd.read_excel(path, sheet_name="PGVector_Info"),
    }


def candidate_preview(candidates_df: pd.DataFrame, candidate_id: int) -> str:
    row = candidates_df.loc[candidates_df["CANDIDATE_ID"] == candidate_id]
    if row.empty:
        return ""
    text = str(row.iloc[0]["CV_text"])
    return text[:2000] + ("..." if len(text) > 2000 else "")


def snippet(text: str, limit: int = 240) -> str:
    text = "" if text is None else str(text)
    return text[:limit] + ("..." if len(text) > limit else "")


def job_post_lookup(jobs_df: pd.DataFrame) -> dict:
    return dict(zip(jobs_df["Offer_ID"].astype(str), jobs_df["Job_post_eng"].astype(str)))


def extract_overlap_terms(candidate_text: str, job_text: str, top_k: int = 8) -> list[str]:
    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "have", "has", "are", "you", "your", "will", "our",
        "job", "role", "work", "working", "experience", "candidate", "position", "company", "looking", "required",
        "skills", "good", "full", "time", "part", "offer", "office", "use", "using", "ability", "level",
    }
    c_tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{2,}", candidate_text.lower())
    j_tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{2,}", job_text.lower())
    c_set = {t for t in c_tokens if t not in stopwords}
    j_set = {t for t in j_tokens if t not in stopwords}
    overlap = sorted(c_set & j_set, key=lambda x: (-len(x), x))
    return overlap[:top_k]


def highlight_terms(text: str, terms: list[str]) -> str:
    escaped = html.escape("" if text is None else str(text))
    for term in sorted(set(terms), key=len, reverse=True):
        if not term.strip():
            continue
        pattern = re.compile(rf"(?i)\b({re.escape(term)})\b")
        escaped = pattern.sub(r"<mark>\1</mark>", escaped)
    return escaped.replace("\n", "<br>")


def enrich_matches(df: pd.DataFrame, candidates_df: pd.DataFrame, jobs_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    candidate_map = dict(zip(candidates_df["CANDIDATE_ID"].astype(int), candidates_df["CV_text"].astype(str)))
    job_map = dict(zip(jobs_df["Offer_ID"].astype(str), jobs_df["Job_post_eng"].astype(str)))
    out["CV_preview"] = out["CANDIDATE_ID"].astype(int).map(lambda x: snippet(candidate_map.get(x, "")))
    out["Job_requirements_preview"] = out["Offer_ID"].astype(str).map(lambda x: snippet(job_map.get(x, "")))
    return out


def render_candidate_match_table(df: pd.DataFrame, score_col: str) -> None:
    show_cols = [
        col
        for col in [
            "rank_for_candidate",
            "Offer_ID",
            score_col,
            "Job_title_eng",
            "CV_preview",
            "Job_requirements_preview",
            "match_reason",
        ]
        if col in df.columns
    ]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)


def render_top_pick_card(title: str, df: pd.DataFrame, score_col: str) -> None:
    st.markdown(f"**{title}**")
    if df.empty:
        st.info("No result available.")
        return
    row = df.iloc[0]
    st.metric("Top score", f"{float(row[score_col]):.2f}")
    st.write(f"**Job title:** {row['Job_title_eng']}")
    st.write(f"**Offer ID:** {row['Offer_ID']}")
    if "match_reason" in row and pd.notna(row["match_reason"]):
        st.caption(str(row["match_reason"]))


def main() -> None:
    st.title("CV / job matching demo")

    with st.sidebar:
        st.header("Data Sources")
        matching_path = st.text_input("Matching workbook", value=str(DEFAULT_MATCHING_FILE))
        vector_path = st.text_input("Vector workbook", value=str(DEFAULT_VECTOR_FILE))
        pg_vector_path = st.text_input("Postgres vector workbook", value=str(DEFAULT_PG_VECTOR_FILE))

    matching_exists = Path(matching_path).exists()
    vector_exists = Path(vector_path).exists()
    pg_vector_exists = Path(pg_vector_path).exists()

    if not matching_exists:
        st.error(f"Matching workbook not found: {matching_path}")
        st.stop()

    matching = load_matching_data(matching_path)
    vector = load_vector_data(vector_path) if vector_exists else None
    pg_vector = load_pg_vector_data(pg_vector_path) if pg_vector_exists else None
    matching["top5"] = enrich_matches(matching["top5"], matching["candidates"], matching["jobs"])
    matching["top20"] = enrich_matches(matching["top20"], matching["candidates"], matching["jobs"])
    matching["job_sorted"] = enrich_matches(matching["job_sorted"], matching["candidates"], matching["jobs"])
    if vector is not None:
        vector["top"] = enrich_matches(vector["top"], matching["candidates"], matching["jobs"])
        vector["job_sorted"] = enrich_matches(vector["job_sorted"], matching["candidates"], matching["jobs"])
    if pg_vector is not None:
        pg_vector["top"] = enrich_matches(pg_vector["top"], matching["candidates"], matching["jobs"])
        pg_vector["job_sorted"] = enrich_matches(pg_vector["job_sorted"], matching["candidates"], matching["jobs"])
    jobs_map = job_post_lookup(matching["jobs"])

    candidate_ids = sorted(matching["top5"]["CANDIDATE_ID"].dropna().astype(int).unique().tolist())
    selected_candidate = st.sidebar.selectbox("Candidate ID", candidate_ids)

    st.markdown("### Project Overview")
    st.write(
        "This demo shows how we match candidate CVs to job posts using three approaches: "
        "a ranking-based matching engine, a local vector retrieval pipeline, and a PostgreSQL + pgvector search pipeline."
    )

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Candidates", len(matching["candidates"]))
    metric2.metric("Jobs", len(matching["jobs"]))
    metric3.metric("Matching outputs", len(matching["top20"]))
    metric4.metric("Selected candidate", int(selected_candidate))

    overview_left, overview_mid, overview_right = st.columns(3)
    matching_top = matching["top5"].loc[matching["top5"]["CANDIDATE_ID"] == selected_candidate].copy()
    with overview_left:
        render_top_pick_card("Matching engine top recommendation", matching_top, "affinity_score")
    with overview_mid:
        if vector is None:
            st.markdown("**Local vector retrieval**")
            st.info("Workbook not available.")
        else:
            vector_top_overview = vector["top"].loc[vector["top"]["CANDIDATE_ID"] == selected_candidate].copy()
            render_top_pick_card("Local vector top recommendation", vector_top_overview, "vector_similarity_score")
    with overview_right:
        if pg_vector is None:
            st.markdown("**PostgreSQL vector search**")
            st.info("Workbook not available.")
        else:
            pg_top_overview = pg_vector["top"].loc[pg_vector["top"]["CANDIDATE_ID"] == selected_candidate].copy()
            render_top_pick_card("PostgreSQL top recommendation", pg_top_overview, "vector_similarity_score")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Candidate Top Matches",
            "Job Title View",
            "Vector Retrieval",
            "Postgres Vector",
            "Compare All",
        ]
    )

    with tab1:
        st.subheader(f"Candidate {selected_candidate}: Best Matching Jobs")
        left, right = st.columns([1.2, 1.8])
        with left:
            st.markdown("**CV Preview**")
            st.text_area(
                "Candidate CV",
                candidate_preview(matching["candidates"], int(selected_candidate)),
                height=420,
                label_visibility="collapsed",
            )
        with right:
            candidate_top5 = matching["top5"].loc[matching["top5"]["CANDIDATE_ID"] == selected_candidate].copy()
            candidate_top20 = matching["top20"].loc[matching["top20"]["CANDIDATE_ID"] == selected_candidate].copy()
            st.markdown("**Top 5 with reasons**")
            render_candidate_match_table(candidate_top5, "affinity_score")
            if not candidate_top5.empty:
                selected_offer = st.selectbox(
                    "Choose one recommended job to inspect in detail",
                    candidate_top5["Offer_ID"].astype(str).tolist(),
                )
                selected_row = candidate_top5.loc[candidate_top5["Offer_ID"].astype(str) == selected_offer].iloc[0]
                selected_job_text = jobs_map.get(str(selected_offer), "")
                selected_cv_text = candidate_preview(matching["candidates"], int(selected_candidate))
                overlap_terms = extract_overlap_terms(selected_cv_text, selected_job_text)

                detail_left, detail_right = st.columns(2)
                with detail_left:
                    st.markdown("**Candidate CV with highlighted overlap**")
                    st.markdown(
                        f"<div style='padding:12px;border:1px solid #ddd;border-radius:8px;max-height:420px;overflow:auto;'>{highlight_terms(selected_cv_text, overlap_terms)}</div>",
                        unsafe_allow_html=True,
                    )
                with detail_right:
                    st.markdown("**Job requirements / description with highlighted overlap**")
                    st.markdown(
                        f"<div style='padding:12px;border:1px solid #ddd;border-radius:8px;max-height:420px;overflow:auto;'>{highlight_terms(selected_job_text, overlap_terms)}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("**Selected match explanation**")
                st.write(selected_row["match_reason"])
            with st.expander("Show top 20 ranking"):
                render_candidate_match_table(candidate_top20, "affinity_score")

    with tab2:
        st.subheader("Top-5 Matches Sorted by Job Title")
        job_titles = sorted(matching["job_sorted"]["Job_title_eng"].dropna().astype(str).unique().tolist())
        selected_title = st.selectbox("Job title", job_titles)
        job_slice = matching["job_sorted"].loc[matching["job_sorted"]["Job_title_eng"] == selected_title].copy()
        st.metric("Candidates shown", len(job_slice))
        st.dataframe(job_slice, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Vector Database Retrieval")
        if vector is None:
            st.warning("Vector workbook not found. Generate Matching_validation_vector_db.xlsx first.")
        else:
            st.markdown("**Vector store info**")
            st.dataframe(vector["info"], use_container_width=True, hide_index=True)
            vector_top = vector["top"].loc[vector["top"]["CANDIDATE_ID"] == selected_candidate].copy()
            st.markdown(f"**Candidate {selected_candidate}: vector-based top jobs**")
            render_candidate_match_table(vector_top, "vector_similarity_score")

    with tab4:
        st.subheader("PostgreSQL Vector Search")
        if pg_vector is None:
            st.warning("PostgreSQL vector workbook not found. Generate Matching_validation_postgres_vector.xlsx first.")
        else:
            st.markdown("**PostgreSQL vector store info**")
            st.dataframe(pg_vector["info"], use_container_width=True, hide_index=True)
            pg_top = pg_vector["top"].loc[pg_vector["top"]["CANDIDATE_ID"] == selected_candidate].copy()
            st.markdown(f"**Candidate {selected_candidate}: PostgreSQL vector-search top jobs**")
            render_candidate_match_table(pg_top, "vector_similarity_score")

    with tab5:
        st.subheader("Matching Engine vs Local Vector vs PostgreSQL Vector")
        st.write(
            "Top 5 from each method for the same candidate."
        )
        expl1, expl2, expl3 = st.columns(3)
        with expl1:
            st.caption("Matching engine: combines keyword overlap, title similarity, and semantic features.")
        with expl2:
            st.caption("Local vector retrieval: focuses on semantic closeness in the local embedding space.")
        with expl3:
            st.caption("PostgreSQL vector search: runs nearest-neighbor retrieval through pgvector in the database.")

        matching_top = matching["top5"].loc[matching["top5"]["CANDIDATE_ID"] == selected_candidate].copy()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Matching engine**")
            render_candidate_match_table(matching_top, "affinity_score")

        with col2:
            if vector is None:
                st.warning("Local vector workbook missing.")
            else:
                vector_top = vector["top"].loc[vector["top"]["CANDIDATE_ID"] == selected_candidate].copy()
                st.markdown("**Local vector retrieval**")
                render_candidate_match_table(vector_top, "vector_similarity_score")

        with col3:
            if pg_vector is None:
                st.warning("PostgreSQL vector workbook missing.")
            else:
                pg_top = pg_vector["top"].loc[pg_vector["top"]["CANDIDATE_ID"] == selected_candidate].copy()
                st.markdown("**PostgreSQL vector search**")
                render_candidate_match_table(pg_top, "vector_similarity_score")

        if vector is not None and pg_vector is not None:
            vector_top = vector["top"].loc[vector["top"]["CANDIDATE_ID"] == selected_candidate].copy()
            pg_top = pg_vector["top"].loc[pg_vector["top"]["CANDIDATE_ID"] == selected_candidate].copy()
            overlap_local = set(matching_top["Offer_ID"].astype(str)) & set(vector_top["Offer_ID"].astype(str))
            overlap_pg = set(matching_top["Offer_ID"].astype(str)) & set(pg_top["Offer_ID"].astype(str))
            overlap_vector_pg = set(vector_top["Offer_ID"].astype(str)) & set(pg_top["Offer_ID"].astype(str))
            common_all = (
                set(matching_top["Offer_ID"].astype(str))
                & set(vector_top["Offer_ID"].astype(str))
                & set(pg_top["Offer_ID"].astype(str))
            )
            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric("Common jobs: Matching + Local", len(overlap_local))
            metric2.metric("Common jobs: Matching + PostgreSQL", len(overlap_pg))
            metric3.metric("Common jobs: Local + PostgreSQL", len(overlap_vector_pg))
            metric4.metric("Common jobs across all 3", len(common_all))

            st.caption("How many of the top 5 jobs the methods have in common. Not an accuracy measure.")

            shared_rows = []
            all_jobs = sorted(
                set(matching_top["Offer_ID"].astype(str))
                | set(vector_top["Offer_ID"].astype(str))
                | set(pg_top["Offer_ID"].astype(str))
            )
            for offer_id in all_jobs:
                shared_rows.append(
                    {
                        "Offer_ID": offer_id,
                        "In Matching": offer_id in set(matching_top["Offer_ID"].astype(str)),
                        "In Local Vector": offer_id in set(vector_top["Offer_ID"].astype(str)),
                        "In PostgreSQL Vector": offer_id in set(pg_top["Offer_ID"].astype(str)),
                    }
                )
            st.markdown("**Shared recommendation map**")
            st.dataframe(pd.DataFrame(shared_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
