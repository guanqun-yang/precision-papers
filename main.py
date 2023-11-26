import logging
import os
import pathlib
import pickle
import random
import sqlite3
import sys
import time
import zlib
from datetime import datetime
from random import shuffle

import numpy as np
from pyzotero import zotero
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sqlitedict import SqliteDict
from tqdm import trange

from setting import setting
from utils.arxiv import (
    get_response,
    parse_response
)
from utils.slack import (
    push_to_slack
)

##################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
    datefmt="%Y-%d-%m-%I-%M-%S"
)

##################################################
class CompressedSqliteDict(SqliteDict):
    """ overrides the encode/decode methods to use zlib, so we get compressed storage """

    def __init__(self, *args, **kwargs):

        def encode(obj):
            return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

        def decode(obj):
            return pickle.loads(zlib.decompress(bytes(obj)))

        super().__init__(*args, **kwargs, encode=encode, decode=decode)

def get_papers_db(flag='r', autocommit=True):
    assert flag in ['r', 'c']
    pdb = CompressedSqliteDict("data/papers.db", tablename='papers', flag=flag, autocommit=autocommit)
    return pdb

def get_metas_db(flag='r', autocommit=True):
    assert flag in ['r', 'c']
    mdb = SqliteDict("data/papers.db", tablename='metas', flag=flag, autocommit=autocommit)
    return mdb


##################################################
# create markdown page
def render_paper(paper_entry: dict, idx: int) -> str:
    # get the arxiv id
    arxiv_id = paper_entry["arxiv_id"]
    # get the title
    title = paper_entry["title"].replace("\n", " ")
    # get the arxiv url
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    # get the abstract
    abstract = paper_entry["abstract"]
    # get the authors
    authors = paper_entry["authors"]
    paper_string = f'## {idx}. [{title}]({arxiv_url}) <a name="link{idx}"></a>\n'
    # paper_string = f'## {idx}. [{title}]({arxiv_url}) <a id="link{idx}"></a>\n'
    paper_string += f"**ArXiv ID:** {arxiv_id}\n"
    paper_string += f'**Authors:** {", ".join(authors)}\n\n'
    paper_string += f"**Abstract:** {abstract}\n\n"

    return paper_string + "\n---\n"


def render_title_and_author(paper_entry: dict, idx: int) -> str:
    title = paper_entry["title"].replace("\n", " ")
    authors = paper_entry["authors"]
    paper_string = f"{idx}. [{title}](#link{idx})\n"
    paper_string += f'**Authors:** {", ".join(authors)}\n'
    return paper_string


def render_md_string(papers):
    output_string = (
        "# Personalized Daily Arxiv Papers "
        + datetime.today().strftime("%m/%d/%Y")
        + "\nTotal relevant papers: "
        + str(len(papers))
        + "\n\n"
        + "Table of contents with paper titles:\n\n"
    )
    title_strings = [
        render_title_and_author(paper, i)
        for i, paper in enumerate(papers)
    ]
    output_string = output_string + "\n".join(title_strings) + "\n---\n"
    # render each paper
    paper_strings = [
        render_paper(paper, i) for i, paper in enumerate(papers)
    ]
    # join all papers into one string
    output_string = output_string + "\n".join(paper_strings)

    return output_string


##################################################
pdb = get_papers_db(flag='c')
mdb = get_metas_db(flag='c')

# fetch up to 100 papers for 20 times, leading up to 2000 papers in total
# stop fetching papers when the timestamp is beyond 24 hours of UTC 0:00 today

current_time = time.time()
search_query = 'cat:cs.LG+OR+cat:cs.CL+OR+cat:cs.AI+OR+cat:cs.NE'

n_papers = len(pdb)

for k in trange(0, 2100, 100):
    response = get_response(
        search_query=search_query,
        start_index=k,
    )

    time.sleep(1 + random.uniform(1, 4))

    # after parsing, there will be some new fields, including "_time", "_id", "_idv"
    items = parse_response(
        response,
    )

    if current_time - items[0]["_time"] > setting.time_delta:
        message = f"THERE ARE NO PAPERS IN THE PAST {setting.time_delta} SECONDS FOR QUERY={search_query}"
        logging.info(message)
        break

    for item in items:
        pdb[item['_id']] = item
        mdb[item['_id']] = {'_time': item['_time']}

##################################################

n_new_papers = len(pdb) - n_papers
logging.info(f"NUMBER OF NEW PAPERS: {n_new_papers}\n")

if n_new_papers == 0 and not setting.debug:
    push_to_slack(
        [
            {
                "title": f"NO NEW PAPER IN PAST {setting.time_delta} SECONDS",
                "abstract": "",
                "arxiv_id": f"QUERY={search_query}",
                "authors": [""],
            }
        ]
    )
    sys.exit()

##################################################
# used for TF-IDF feature extraction
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
# Zotero
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
positive_dict = {
    "x": x_positive
}
candidate_dict = {
    'pids': list(pdb.keys()),
    'x': x_candidate,
    'vocab': feature_extractor.vocabulary_,
    'idf': feature_extractor._tfidf.idf_,
}

##################################################
# caching features to pickle files

# save_positives(positive_dict)
# save_features(candidate_dict)

##################################################

X_positive = positive_dict["x"]
X_candidate = candidate_dict["x"]

n_positive = X_positive.shape[0]
n_candidate = X_candidate.shape[0]

X = sparse.vstack([X_positive, X_candidate])
y = np.concatenate([
    np.ones(n_positive),
    np.zeros(n_candidate)
])

# create recommendation
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
for i, pid in enumerate(sorted_pids[:setting.topk]):
    item = pdb[pid]

    papers.append(
        {
            "title": " ".join(item["title"].split()),
            "abstract": item["summary"],
            "authors": [p["name"] for p in item["authors"]],
            "arxiv_id": item["_id"]
        }
    )

    if (i + 1) % 50 == 0:
        push_to_slack(papers[-50:])

# push remaining papers
push_to_slack(papers[-(len(papers) % 50):])

# update output.md
markdown_string = render_md_string(papers)
pathlib.Path("data/output.md").write_text(markdown_string)


