import numpy as np

from scipy import sparse
from sklearn.svm import LinearSVC
from utils.db import (
    get_papers_db,
    load_features,
    load_positives,
)
from utils.slack import (
    Paper,
    push_to_slack
)
from setting import setting

pdb = get_papers_db("r")

positive_dict = load_positives()
candidate_dict = load_features()

X_positive = positive_dict["x"]
X_candidate = candidate_dict["x"]

n_positive = X_positive.shape[0]
n_candidate = X_candidate.shape[0]

X = sparse.vstack([X_positive, X_candidate])
y = np.concatenate([
    np.ones(n_positive),
    np.zeros(n_candidate)
])

# goal: using the vectorized data X and y (rather than y_true) to get y_pred as close as y_true
clf = LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.01)
clf.fit(X, y)

scores = clf.decision_function(X)
sorted_pids = [
    candidate_dict["pids"][idx-n_positive] for idx in np.argsort(scores)[::-1]
    if idx not in np.arange(n_positive)
]

# return the top-k papers
# the records have to have "arxiv_id", "abstract", "authors", "title" columns
papers = list()
for pid in sorted_pids[:setting.topk]:
    item = pdb[pid]

    papers.append(
        {
            "title": item["title"],
            "abstract": item["summary"],
            "authors": [p["name"] for p in item["authors"]],
            "arxiv_id": item["_id"]
        }
    )

push_to_slack(papers)

