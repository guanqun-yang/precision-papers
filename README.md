# PrecisionPapers: Paper Recommendation System

## Overview

The library automatically fetches the user data stored in one of the Zotero databases (for example, this [one](https://www.zotero.org/guanqun.yang/collections/FS6NV5KD) named `FS6NV5KD`) and recommends new papers uploaded to arXiv based on the user's preference. The recommendation results will be sent to the user through a Slack message.

The underlying algorithm is the same as arXiv-sanity, which is powerful enough so that statistically, the crowd decisions with arXix-sanity (mostly) agree with the ICLR 2017 expert decisions (see Anderj Kaparthy's post [here](https://karpathy.medium.com/iclr-2017-vs-arxiv-sanity-d1488ac5c131)).

## Dependencies

```
conda create --name precision-papers python==3.9

conda install ipython
pip install scikit-learn

# special dependencies
pip install pyzotero sqlitedict
```

## Key Design Choices

-   The feature extraction is done only once for each unique paper. The extracted features are stored in a database for future use.

-   The core recommendation algorithm is TF-IDF features and an `svm.LinearSVC` classifier that outputs a scalar through `clf.decision_funcion(X)`. Specifically, all of the papers are mixed; the ones that are known to be relevant are tagged `1,` and others are tagged `0`. Then, the papers that are ranked at the top in `sortix` will be recommended.

    For example, the following is the MWE using the classical dataset with 2 topics. We randomly mask 50% of the samples to 0 (i.e., "unknown") and see if the top 50 documents (which are supposed to be "relevant") are indeed relevant. Despite the simplicity, this algorithm works surprisingly well with an AUROC of more than 0.95, and **all** top-50 documents are indeed "relevant."

    In the setting of recommending academic papers, they will be the initial set of papers that we are already interested in, and our goal is to find the ones that should also be tagged `1` in the incoming stream of papers from arXiv every day.

    ```python
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
    ```

    
