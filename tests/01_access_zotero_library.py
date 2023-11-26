import os

import pandas as pd
from pyzotero import zotero

library_id = "9338796"
library_type = "user"
api_key = os.getenv("ZOTERO_API_KEY")

# initialize client
zot = zotero.Zotero(library_id, library_type, api_key)


collection_id = "FS6NV5KD"
items = [
    item for item in zot.collection_items(collection_id)
    if item["data"]["itemType"] not in ["attachment", "note"]
]

records = list()
for item in items:
    records.append(
        {
            "title": item["data"]["title"],
            "abstract": item["data"]["abstractNote"],
            "author": ", ".join(["{} {}".format(p["firstName"], p["lastName"]) for p in item["data"]["creators"]]),
            "url": item["data"]["url"],
        }
    )

df = pd.DataFrame(records)




# items = zot.top(limit=5)
# # we've retrieved the latest five top-level items in our library
# # we can print each item's item type and ID
# for item in items:
#     print('Item: %s | Key: %s' % (item['data']['itemType'], item['data']['key']))