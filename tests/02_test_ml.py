import time
import random
import logging
import pandas as pd

from tqdm import trange
from aslite.arxiv import (
    get_response,
    parse_response
)

from sklearn.svm import SVC
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
    datefmt="%Y-%d-%m-%I-%M-%S"
)

# fetch up to 100 papers for 20 times, leading up to 2000 papers in total
# stop fetching papers when the timestamp is beyond 24 hours of UTC 0:00 today
search_query = 'cat:cs.LG+OR+cat:cs.CL+OR+cat:cs.AI+OR+cat:cs.NE'

records = list()
current_time = time.time()

for k in trange(0, 2100, 100):
    response = get_response(
        search_query=search_query,
        start_index=k,
    )

    items = parse_response(
        response,
    )

    # only fetch papers in the past one day (86400 seconds)
    if current_time - items[0]["_time"] > 86400:
        logging.info(f"THERE ARE NO PAPERS IN THE PAST 24 HOURS FOR QUERY={search_query}")
        break

    for item in items:
        records.append(
            {
                "number": item["_id"],
                "title": item["title"],
                "abstract": item["summary"],
                "author": ", ".join(author["name"] for author in item["authors"]),
                "time": item["_time"],
                "url": item["link"]
            }
        )

paper_df = pd.DataFrame(records)