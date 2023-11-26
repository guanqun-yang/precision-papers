import logging
import os
from random import shuffle

import numpy as np
from pyzotero import zotero
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.db import (
    get_papers_db,
    get_metas_db,
    save_features,
    save_positives,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
    datefmt="%Y-%d-%m-%I-%M-%S"
)

##################################################

pdb = get_papers_db(flag='r')
mdb = get_metas_db(flag='r')

def make_corpus(training: bool):
    assert isinstance(training, bool)

    # determine which papers we will use to build tfidf
    if training:
        # crop to a random subset of papers
        keys = list(pdb.keys())
        shuffle(keys)
    else:
        keys = pdb.keys()

    # yield the abstracts of the papers
    for p in keys:
        d = pdb[p]
        author_str = ' '.join([a['name'] for a in d['authors']])
        yield ' '.join([d['title'], d['summary'], author_str])

##################################################

# initialize client
zot = zotero.Zotero(
    library_id="9338796",
    library_type="user",
    api_key=os.getenv("ZOTERO_API_KEY")
)
def retrieve_relevant_papers():
    collection_id = "FS6NV5KD"
    items = [
        item for item in zot.collection_items(collection_id)
        if item["data"]["itemType"] not in ["attachment", "note"]
    ]

    for item in items:
        title = item["data"]["title"]
        abstract = item["data"]["abstractNote"]
        author = " ".join(["{} {}".format(p["firstName"], p["lastName"]) for p in item["data"]["creators"]])

        yield " ".join([title, abstract, author])

##################################################

feature_extractor = TfidfVectorizer(
    input='content',
    encoding='utf-8',
    decode_error='replace',
    strip_accents='unicode',
    lowercase=True,
    analyzer='word',
    stop_words='english',
    token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
    ngram_range=(1, 2),
    max_features=20000,
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True,
    max_df=5,
    min_df=1
)

# note that as there are enough papers in the database
# we do not worry about whether the fitted and transformed corpus are same or not
logging.info("Fitting feature extractor....")
feature_extractor.fit(make_corpus(training=True))

logging.info("Extracting features...")

x_candidate = feature_extractor.transform(make_corpus(training=False)).astype(np.float32)
x_positive = feature_extractor.transform(retrieve_relevant_papers()).astype(np.float32)

logging.info("Saving features to disk...")
positive_features = {
    "x": x_positive
}
candidate_features = {
    'pids': list(pdb.keys()),
    'x': x_candidate,
    'vocab': feature_extractor.vocabulary_,
    'idf': feature_extractor._tfidf.idf_,
}

save_positives(positive_features)
save_features(candidate_features)
