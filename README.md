# LRec

# Dependencies

    + envoy
    + progressbar
    + sklearn 0.17
    + Cython

# Installation

    python setup.py install

# Data Input

    The data format is tab separated user-item-rating score
        <user>\t<item>\t<rating>
    For one class collaborative filtering, the rating is 1
    Also, the different delimiter can be used by defining a custom parser or passind delim in UserItemRatingParser.

# Running the code

    Please refer lrec.ipynb for step by step guide to run the code. The sample dataset(lastFM) is included in the data folder.



