import os
from pyzotero import zotero

library_id = "9338796"
library_type = "user"
api_key = os.getenv("ZOTERO_API_KEY")

zot = zotero.Zotero(library_id, library_type, api_key)
items = zot.top(limit=5)
# we've retrieved the latest five top-level items in our library
# we can print each item's item type and ID
for item in items:
    print('Item: %s | Key: %s' % (item['data']['itemType'], item['data']['key']))