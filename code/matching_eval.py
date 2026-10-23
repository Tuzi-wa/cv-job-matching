import argparse
import hashlib
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import matching_app as m

SEED = 20240517


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the ranking on candidates that were not used to tune the weights.")
    p.add_argument("--train", required=True, help="Path to Matching_training.xlsx")
    p.add_argument("--embeddings", choices=["svd", "sentence", "both"], default="svd",
                   help="svd: TF-IDF + TruncatedSVD, sentence: sentence-transformer embeddings, both: use both")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="sentence-transformers model")
    p.add_argument("--device", default=None, help="cpu, mps or cuda (default: picked by sentence-transformers)")
    p.add_argument("--max-seq-length", type=int, default=512, help="longer texts are truncated")
    p.add_argument("--cache-dir", default="../results/embedding_cache",
                   help="where encoded texts are saved, so the model only runs once per text set")
    p.add_argument("--tuner", choices=["app", "random"], default=None,
                   help="app: the weight search of matching_app.py (svd only), random: random search on MRR. "
                        "Default: app for svd, random otherwise")
    p.add_argument("--tune-frac", type=float, default=0.5, help="share of candidates used to tune the weights")
    return p.parse_args()


def build_features(tc, tj, mode, args):
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
        encode = Encoder(args.model, args.max_seq_length, args.cache_dir, args.device)
        # title once plus the post; job_document repeats the title for TF-IDF, which is not needed here
        job_texts = [f"{t}. {b}" for t, b in zip(tj["job_title_clean"], tj["job_post_clean"])]
        cand_vec = encode(list(tc["cv_clean"]))
        job_vec = encode(job_texts)
        title_vec = encode(list(tj["job_title_clean"]))
        feats += [
            np.clip(cand_vec @ job_vec.T, 0.0, 1.0).astype(np.float32),
            np.clip(cand_vec @ title_vec.T, 0.0, 1.0).astype(np.float32),
        ]
        names += ["sent_body", "sent_title"]

    return feats, names


class Encoder:
    """Encodes texts with a sentence-transformers model and caches the vectors on disk."""

    def __init__(self, model_name, max_seq_length, cache_dir, device=None):
        self.device = device
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.cache_dir = Path(cache_dir)
        self.model = None

    def __call__(self, texts):
        key = hashlib.sha1(
            "\x00".join([self.model_name, str(self.max_seq_length)] + list(texts)).encode("utf-8")
        ).hexdigest()[:16]
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            return np.load(path)
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            print(f"loading {self.model_name} ...", flush=True)
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model.max_seq_length = self.max_seq_length
        print(f"encoding {len(texts)} texts", flush=True)
        vectors = self.model.encode(list(texts), batch_size=32, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors


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
    rr = []
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
        rr.append(1.0 / pos)
    k = np.mean([len(gold[r]) for r in rows if gold[r]])
    base1, base10 = k / n_jobs, min(1.0, 10 * k / n_jobs)
    if not verbose:
        return dict(r1=r1 / n, r5=r5 / n, r10=r10 / n, mrr=mrr / n, rr=np.array(rr))
    print(f"\n{label}: n={n}, {k:.2f} correct jobs per candidate on average")
    print(f"  Recall@1 {r1/n:.4f}   Recall@5 {r5/n:.4f}   Recall@10 {r10/n:.4f}   MRR {mrr/n:.4f}")
    print(f"  random baseline R@1 {base1:.5f}  R@10 {base10:.5f}"
          f"  (lift R@1 {(r1/n)/base1:.1f}x, R@10 {(r10/n)/base10:.1f}x)")
    return dict(r1=r1 / n, r5=r5 / n, r10=r10 / n, mrr=mrr / n, rr=np.array(rr))


def bootstrap_diff(a, b, n_boot=2000):
    """Mean of a - b and a 95% interval from resampling the evaluation candidates."""
    d = a - b
    idx = np.random.default_rng(SEED).integers(0, len(d), (n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return d.mean(), lo, hi


def main() -> int:
    args = parse_args()
    training = m.load_training_dataset(os.path.abspath(args.train))
    tc = m.prepare_candidates(training.candidates)
    tj = m.prepare_jobs(training.jobs)

    print(f"candidates {len(tc)} | jobs {len(tj)} | labelled matches {len(training.matches)}")
    print(f"building features (embeddings={args.embeddings}) ...")
    feats, names = build_features(tc, tj, args.embeddings, args)
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

    tuner = args.tuner or ("app" if args.embeddings == "svd" else "random")
    if tuner == "app" and args.embeddings != "svd":
        raise SystemExit("--tuner app only works with --embeddings svd")
    if tuner == "app":
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

    combined = {
        "strict": score(feats, weights, eval_rows, strict, n_jobs, "strict (only the highest match_score counts)"),
        "lenient": score(feats, weights, eval_rows, rel, n_jobs, "lenient (any labelled match counts)"),
    }

    print("\neach feature on its own (strict / lenient):")
    for i, name in enumerate(names):
        single = np.zeros(len(feats), np.float32)
        single[i] = 1.0
        s = score(feats, single, eval_rows, strict, n_jobs, name, verbose=False)
        l = score(feats, single, eval_rows, rel, n_jobs, name, verbose=False)
        print(f"  {name:10s} R@10 {s['r10']:.4f} / {l['r10']:.4f}   MRR {s['mrr']:.4f} / {l['mrr']:.4f}")
        if name == "word":
            word_only = {"strict": s, "lenient": l}

    if "word" in names:
        print("\nMRR of the combination minus word TF-IDF alone (95% bootstrap interval):")
        for gold_name in ("strict", "lenient"):
            d, lo, hi = bootstrap_diff(combined[gold_name]["rr"], word_only[gold_name]["rr"])
            print(f"  {gold_name:8s} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
