import argparse
import os
import random
from collections import defaultdict

import numpy as np

import matching_app as m

SEED = 20240517


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the matching ranker with a leakage-free split.")
    p.add_argument("--train", required=True, help="Path to Matching_training.xlsx")
    p.add_argument("--embeddings", choices=["svd", "sentence", "both"], default="svd",
                   help="svd: TF-IDF + TruncatedSVD, sentence: sentence-transformer embeddings, both: use both")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--tune-frac", type=float, default=0.5, help="share of candidates used to tune the weights")
    return p.parse_args()


def build_features(tc, tj, mode, model_name):
    feats, names = [], []

    if mode in ("svd", "both"):
        wv, cvv, tv = m.fit_vectorizers(
            all_candidate_docs=list(tc["cv_clean"]),
            all_job_docs=list(tj["job_document"]),
            all_titles=list(tj["job_title_clean"]),
        )
        c_w = m.transform_texts(wv, tc["cv_clean"])
        c_c = m.transform_texts(cvv, tc["cv_clean"])
        c_t = m.transform_texts(tv, tc["cv_clean"])
        j_w = m.transform_texts(wv, tj["job_document"])
        j_c = m.transform_texts(cvv, tj["job_document"])
        j_t = m.transform_texts(tv, tj["job_title_clean"])

        svd_main, svd_title = m.fit_svd_embeddings(c_w, j_w, c_t, j_t)
        c_me, j_me = m.transform_embeddings(svd_main, c_w), m.transform_embeddings(svd_main, j_w)
        c_te, j_te = m.transform_embeddings(svd_title, c_t), m.transform_embeddings(svd_title, j_t)

        feats += [
            m.linear_kernel(c_w, j_w).astype(np.float32),
            m.linear_kernel(c_c, j_c).astype(np.float32),
            m.linear_kernel(c_t, j_t).astype(np.float32),
            np.clip(c_me @ j_me.T, 0.0, 1.0).astype(np.float32),
            np.clip(c_te @ j_te.T, 0.0, 1.0).astype(np.float32),
        ]
        names += ["word", "char", "title", "svd_main", "svd_title"]

    if mode in ("sentence", "both"):
        from sentence_transformers import SentenceTransformer

        print(f"loading {model_name} ...")
        st = SentenceTransformer(model_name)
        cand_vec = st.encode(list(tc["cv_clean"]), batch_size=64, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=True)
        job_vec = st.encode(list(tj["job_document"]), batch_size=64, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=True)
        title_vec = st.encode(list(tj["job_title_clean"]), batch_size=64, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=True)
        feats += [
            np.clip(cand_vec @ job_vec.T, 0.0, 1.0).astype(np.float32),
            np.clip(cand_vec @ title_vec.T, 0.0, 1.0).astype(np.float32),
        ]
        names += ["sent_body", "sent_title"]

    return feats, names


def tune_weights(feats, rows, gold, n_trials=64):
    """Random search for the weights on the tuning candidates (used when sentence embeddings are added)."""
    rng = random.Random(SEED)
    trials = [tuple([1.0] * len(feats))]
    trials += [tuple(round(rng.uniform(0.0, 3.0), 2) for _ in feats) for _ in range(n_trials)]

    best_w, best_mrr = np.ones(len(feats), np.float32), -1.0
    for w in trials:
        wv = np.array(w, np.float32)
        if wv.sum() <= 0:
            continue
        S = sum(a * f for a, f in zip(wv, feats)) / wv.sum()
        mrr = 0.0
        for r in rows:
            order = np.argsort(-S[r])
            hit = np.where(np.isin(order, list(gold[r])))[0]
            if len(hit):
                mrr += 1.0 / (hit[0] + 1)
        mrr /= max(1, len(rows))
        if mrr > best_mrr:
            best_mrr, best_w = mrr, wv
    return best_w, best_mrr


def score(feats, weights, rows, gold, n_jobs, label, verbose=True):
    S = sum(a * f for a, f in zip(weights, feats)) / weights.sum()
    r1 = r5 = r10 = 0
    mrr = 0.0
    n = 0
    for r in rows:
        if not gold[r]:
            continue
        order = np.argsort(-S[r])
        hit = np.where(np.isin(order, list(gold[r])))[0]
        if not len(hit):
            continue
        pos = hit[0] + 1
        n += 1
        r1 += pos <= 1
        r5 += pos <= 5
        r10 += pos <= 10
        mrr += 1.0 / pos
    k = np.mean([len(gold[r]) for r in rows if gold[r]])
    base1, base10 = k / n_jobs, min(1.0, 10 * k / n_jobs)
    if not verbose:
        return dict(r1=r1 / n, r5=r5 / n, r10=r10 / n, mrr=mrr / n)
    print(f"\n{label}: n={n}, {k:.2f} correct jobs per candidate on average")
    print(f"  Recall@1 {r1/n:.4f}   Recall@5 {r5/n:.4f}   Recall@10 {r10/n:.4f}   MRR {mrr/n:.4f}")
    print(f"  random baseline R@1 {base1:.5f}  R@10 {base10:.5f}"
          f"  (lift R@1 {(r1/n)/base1:.1f}x, R@10 {(r10/n)/base10:.1f}x)")
    return dict(r1=r1 / n, r5=r5 / n, r10=r10 / n, mrr=mrr / n)


def main() -> int:
    args = parse_args()
    training = m.load_training_dataset(os.path.abspath(args.train))
    tc = m.prepare_candidates(training.candidates)
    tj = m.prepare_jobs(training.jobs)

    print(f"candidates {len(tc)} | jobs {len(tj)} | labelled matches {len(training.matches)}")
    print(f"building features (embeddings={args.embeddings}) ...")
    feats, names = build_features(tc, tj, args.embeddings, args.model)
    print(f"features: {', '.join(names)}")

    cidx = {int(c): i for i, c in enumerate(tc["CANDIDATE_ID"].astype(int))}
    jidx = {str(j): i for i, j in enumerate(tj["Offer_ID"].astype(str))}
    n_jobs = len(jidx)

    # all labelled matches per candidate, and only the highest-scored one
    rel = defaultdict(set)
    for r in training.matches.itertuples(index=False):
        if int(r.CANDIDATE_ID) in cidx and str(r.Offer_ID) in jidx:
            rel[cidx[int(r.CANDIDATE_ID)]].add(jidx[str(r.Offer_ID)])
    top1 = (training.matches.sort_values(["CANDIDATE_ID", "match_score"], ascending=[True, False])
            .groupby("CANDIDATE_ID", group_keys=False).head(1))
    strict = defaultdict(set)
    for r in top1.itertuples(index=False):
        if int(r.CANDIDATE_ID) in cidx and str(r.Offer_ID) in jidx:
            strict[cidx[int(r.CANDIDATE_ID)]].add(jidx[str(r.Offer_ID)])

    # split by candidate: weights are chosen on one part and evaluated on the other
    rows = sorted(rel.keys())
    random.Random(SEED).shuffle(rows)
    cut = int(len(rows) * args.tune_frac)
    tune_rows, eval_rows = rows[:cut], rows[cut:]
    print(f"\ntuning on {len(tune_rows)} candidates, evaluating on {len(eval_rows)} others")

    if args.embeddings == "svd":
        # same weight search as matching_app.py, restricted to the tuning candidates
        tune_ids = {int(tc["CANDIDATE_ID"].iloc[r]) for r in tune_rows}
        tune_matches = training.matches[training.matches["CANDIDATE_ID"].astype(int).isin(tune_ids)]
        weights = m.tune_rank_weights(
            matches=tune_matches,
            candidate_ids=list(tc["CANDIDATE_ID"].astype(int)),
            job_ids=list(tj["Offer_ID"].astype(str)),
            feature_matrices=feats,
        )
    else:
        weights, _ = tune_weights(feats, tune_rows, strict)
    print("weights: " + ", ".join(f"{n}={w:.2f}" for n, w in zip(names, weights)))

    score(feats, weights, eval_rows, strict, n_jobs, "strict (only the highest match_score counts)")
    score(feats, weights, eval_rows, rel, n_jobs, "lenient (any labelled match counts)")

    print("\neach feature on its own (strict / lenient):")
    for i, name in enumerate(names):
        single = np.zeros(len(feats), np.float32)
        single[i] = 1.0
        s = score(feats, single, eval_rows, strict, n_jobs, name, verbose=False)
        l = score(feats, single, eval_rows, rel, n_jobs, name, verbose=False)
        print(f"  {name:10s} R@10 {s['r10']:.4f} / {l['r10']:.4f}   MRR {s['mrr']:.4f} / {l['mrr']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
