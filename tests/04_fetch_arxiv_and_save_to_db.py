import logging
import random
import time

from tqdm import trange

from utils.arxiv import (
    get_response,
    parse_response
)
from utils.db import (
    get_papers_db,
    get_metas_db,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
    datefmt="%Y-%d-%m-%I-%M-%S"
)

pdb = get_papers_db(flag='c')
mdb = get_metas_db(flag='c')

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

    time.sleep(1 + random.uniform(1, 4))

    # after parsing, there will be some new fields, including "_time", "_id", "_idv"
    items = parse_response(
        response,
    )

    # only fetch papers in the past one day (86400 seconds)
    if current_time - items[0]["_time"] > 86400:
        logging.info(f"THERE ARE NO PAPERS IN THE PAST 24 HOURS FOR QUERY={search_query}")
        break

    for item in items:
        pdb[item['_id']] = item
        mdb[item['_id']] = {'_time': item['_time']}




