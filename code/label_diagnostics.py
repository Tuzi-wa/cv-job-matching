import sys

import numpy as np
from scipy.stats import spearmanr
import matching_app as m

path = sys.argv[1] if len(sys.argv) > 1 else "../data/Matching_training.xlsx"
tr = m.load_training_dataset(path)
tc, tj = m.prepare_candidates(tr.candidates), m.prepare_jobs(tr.jobs)

wv, _, _ = m.fit_vectorizers(list(tc["cv_clean"]), list(tj["job_document"]), list(tj["job_title_clean"]))
S = m.linear_kernel(m.transform_texts(wv, tc["cv_clean"]),
                    m.transform_texts(wv, tj["job_document"])).astype(np.float32)

cidx = {int(c): i for i, c in enumerate(tc["CANDIDATE_ID"].astype(int))}
jidx = {str(j): i for i, j in enumerate(tj["Offer_ID"].astype(str))}
pairs = [(cidx[int(r.CANDIDATE_ID)], jidx[str(r.Offer_ID)], float(r.match_score))
         for r in tr.matches.itertuples(index=False)
         if int(r.CANDIDATE_ID) in cidx and str(r.Offer_ID) in jidx]

lab = np.array([S[i, j] for i, j, _ in pairs])
rng = np.random.default_rng(0)
rnd = S[rng.integers(0, S.shape[0], 20000), rng.integers(0, S.shape[1], 20000)]
sc = np.array([s for _, _, s in pairs])
ranks = np.array([(S[i] > S[i, j]).mean() for i, j, _ in pairs])

print(f"TF-IDF similarity, labelled pairs {lab.mean():.4f} vs random pairs {rnd.mean():.4f} ({lab.mean()/rnd.mean():.2f}x)")
print(f"match_score vs similarity: Pearson {np.corrcoef(sc, lab)[0,1]:+.4f}  Spearman {spearmanr(sc, lab).statistic:+.4f}")
print(f"share of jobs ranked above the labelled job: mean {ranks.mean():.3f}, median {np.median(ranks):.3f} (random 0.5)")
