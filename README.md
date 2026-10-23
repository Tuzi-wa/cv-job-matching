# cv-job-matching

Course assignment. Given a set of CVs and job posts, rank the jobs for each candidate and return the top 5 per candidate.

I tried three approaches:

1. `code/matching_app.py` scores every CV-job pair with five similarities and ranks the jobs by their weighted sum: word TF-IDF (1-2 grams), character TF-IDF (3-5 grams), CV against job title, and the word and title TF-IDF vectors reduced with TruncatedSVD (LSA, 256 and 128 dimensions). The weights are chosen from 38 candidate sets by Recall@5 on the training matches. This is the one used for the submission.
2. `code/vector_db_matcher.py` does cosine nearest-neighbour search on the LSA vectors (text and title concatenated, title weighted 1.5) and saves them locally.
3. `code/postgres_vector_setup.py` + `code/postgres_vector_search.py` store the same vectors in PostgreSQL with pgvector (ivfflat index, cosine distance).

The `match_reason` column in the outputs only lists words that the CV and the job post share. It is not used for scoring.

## Result

`code/matching_eval.py` splits the 999 training candidates in two, picks the weights on one half and evaluates on the other 500 candidates, each ranked against all 1,804 jobs. Strict counts only the labelled job with the highest `match_score`, lenient counts any labelled job (3.4 per candidate on average). Besides the submitted features I also tried sentence embeddings from `BAAI/bge-small-en-v1.5` (CV against job post and against job title).

| features | weights | R@10 strict | MRR strict | R@10 lenient | MRR lenient |
|---|---|---|---|---|---|
| word TF-IDF only | - | 0.102 | 0.048 | 0.194 | 0.104 |
| TF-IDF + LSA (submitted) | search in `matching_app.py` | 0.096 | 0.049 | 0.186 | 0.097 |
| TF-IDF + LSA | random search on MRR | 0.102 | 0.053 | 0.204 | 0.108 |
| sentence embeddings | random search on MRR | 0.060 | 0.036 | 0.146 | 0.079 |
| TF-IDF + LSA + sentence embeddings | random search on MRR | 0.108 | 0.055 | 0.216 | 0.115 |
| random | | 0.006 | | 0.019 | |

Recall@10 is 10-20 times the random baseline. On their own, the sentence embeddings are clearly worse than TF-IDF. Adding them to the TF-IDF features gives the best numbers, but with 500 evaluation candidates the gain over word TF-IDF alone is within noise (lenient MRR +0.011, 95% bootstrap interval -0.005 to +0.028), and the same holds for the other combinations. The labels are only loosely related to the text: `code/label_diagnostics.py` shows that TF-IDF similarity ranks on average 27% of the jobs above the labelled one, and `match_score` is almost uncorrelated with it (Spearman 0.05).

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

The other rows of the table:

```bash
python matching_eval.py --train ../data/Matching_training.xlsx --tuner random
python matching_eval.py --train ../data/Matching_training.xlsx --embeddings sentence
python matching_eval.py --train ../data/Matching_training.xlsx --embeddings both
```

The sentence embeddings need the `sentence-transformers` package. They are saved in `results/embedding_cache/`, so the model only runs once. On a Mac, `--device cpu` was more reliable than the default.

Demo, from the project root after running the scripts above:

```bash
streamlit run demo/streamlit_app.py
```

The demo reads the workbooks from `results/` by default, other paths can be set in the sidebar.

Scores are only meant for ranking within a candidate, not as an absolute measure of fit.
