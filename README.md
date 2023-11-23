# PrecisionPapers: Paper Recommendation System

## Overview

The library automatically fetches the user data stored in one of the Zotero databases (for example, this [one](https://www.zotero.org/guanqun.yang/collections/FS6NV5KD) named `FS6NV5KD`) and recommends new papers uploaded to arXiv based on the user's perference. The recommendation results will be sent to the user through a Slack message.

## Dependencies

```
conda create --name precision-papers python==3.9

conda install ipython
pip install scikit-learn

# special dependencies
pip install pyzotero
```

