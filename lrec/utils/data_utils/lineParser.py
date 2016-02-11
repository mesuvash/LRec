class UserItemRatingParser(object):
    """docstring for userItemRatingParser"""

    def __init__(self, delim="\t"):
        super(UserItemRatingParser, self).__init__()
        self.delim = delim

    def parse(self, line):
        user, item, rating = line.strip().split(self.delim)[:3]
        return (user, item, rating)

