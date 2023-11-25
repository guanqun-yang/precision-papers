import pandas as pd
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

data = fetch_20newsgroups_vectorized()

relevant = "talk.politics.guns"
irrelevant = "rec.sport.baseball"

relevant_index = data.target_names.index(relevant)
irrelevant_index = data.target_names.index(irrelevant)

selected = (data.target == relevant_index) | (data.target == irrelevant_index)

X = data.data[selected]
y = data.target[selected]

y[y == relevant_index] = 1
y[y == irrelevant_index] = 0

# randomly mask 50% of the relevant items
df = pd.DataFrame(
    {
        "y_true": y,
        "y": y,
    }
)
df.loc[df[df.y == 1].sample(frac=0.5).index, "y"] = 0

# goal: using the vectorized data X and y (rather than y_true) to get y_pred as close as y_true
clf = LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.01)
clf.fit(X, y)

scores = clf.decision_function(X)
df["y_proba"] = scores

# overall performance
score = roc_auc_score(y_true=df.y_true.values, y_score=df.y_proba.values)
print(score)

# top-50
sorted_df = df.sort_values("y_proba", ascending=False).iloc[:50]

