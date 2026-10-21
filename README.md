# cv-job-matching

Course assignment. Given a set of CVs and job posts, rank the jobs for each candidate and return the top 5 per candidate.

I tried three approaches:

1. `code/matching_app.py` scores every CV-job pair with five similarities and ranks the jobs by their weighted sum: word TF-IDF (1-2 grams), character TF-IDF (3-5 grams), CV against job title, and the word and title TF-IDF vectors reduced with TruncatedSVD (LSA, 256 and 128 dimensions). The weights are chosen from 38 candidate sets by Recall@5 on the training matches. This is the one used for the submission.
2. `code/vector_db_matcher.py` does cosine nearest-neighbour search on the LSA vectors (text and title concatenated, title weighted 1.5) and saves them locally.
3. `code/postgres_vector_setup.py` + `code/postgres_vector_search.py` store the same vectors in PostgreSQL with pgvector (ivfflat index, cosine distance).

The `match_reason` column in the outputs only lists words that the CV and the job post share. It is not used for scoring.

## Result

`code/matching_eval.py` splits the 999 training candidates in two, picks the weights on one half with the same search as `matching_app.py` and evaluates on the other 500 candidates, each ranked against all 1,804 jobs. Strict counts only the labelled job with the highest `match_score`, lenient counts any labelled job (3.4 per candidate on average).

| | R@10 strict | MRR strict | R@10 lenient | MRR lenient |
|---|---|---|---|---|
| weighted sum | 0.096 | 0.049 | 0.186 | 0.097 |
| word TF-IDF only | 0.102 | 0.048 | 0.194 | 0.104 |
| random | 0.006 | | 0.019 | |

Recall@10 is 10-17 times the random baseline, but the weighted sum does no better than word TF-IDF alone on the held-out candidates. The labels are only loosely related to the text: `code/label_diagnostics.py` shows that TF-IDF similarity ranks on average 27% of the jobs above the labelled one, and `match_score` is almost uncorrelated with it (Spearman 0.05).

## Data

The course data (about 2,300 CVs and 3,800 job posts, with labelled matches for the training set) is not included, because many of the CVs still contain contact details, home addresses and dates of birth. `data/sample_training.xlsx` and `data/sample_validation.xlsx` are small made-up files with the same sheets and columns (`Candidates`, `Job_posts`, `Matches`), so the scripts and the demo can be tried without it. `Template results 2nd Work_filled.xlsx` is the submitted ranking (CANDIDATE_ID, RANK, OFFER_ID).

## Running

Needs pandas, numpy, scikit-learn, scipy, openpyxl, streamlit. The Postgres part also needs a local PostgreSQL with pgvector.

Run from `code/`:

```bash
python matching_app.py --train ../data/sample_training.xlsx --validation ../data/sample_validation.xlsx --output ../results/Matching_validation_scored.xlsx

python matching_eval.py --train ../data/sample_training.xlsx
python label_diagnostics.py ../data/sample_training.xlsx

python vector_db_matcher.py --train ../data/sample_training.xlsx --validation ../data/sample_validation.xlsx --store-dir ../results/vector_db_store --output ../results/Matching_validation_vector_db.xlsx

python postgres_vector_setup.py --train ../data/sample_training.xlsx --validation ../data/sample_validation.xlsx --store-dir ../results/vector_db_store_pg --reset
python postgres_vector_search.py --output ../results/Matching_validation_postgres_vector.xlsx
```

With the course data, replace the sample files with `Matching_training.xlsx` and `Matching_validation.xlsx`.

`matching_eval.py --embeddings sentence` (or `both`) uses sentence-transformer embeddings instead of (or next to) the TF-IDF features and needs the `sentence-transformers` package.

Demo, from the project root after running the scripts above:

```bash
streamlit run demo/streamlit_app.py
```

The demo reads the workbooks from `results/` by default, other paths can be set in the sidebar.

Scores are only meant for ranking within a candidate, not as an absolute measure of fit.
