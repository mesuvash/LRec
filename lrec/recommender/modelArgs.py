
class args(object):
    """docstring for args"""

    def __init__(self):
        super(args, self).__init__()

    def __str__(self):
        fields = []
        for key, value in self.__dict__.items():
            fields.append("%s : %s" % (str(key), str(value)))
        return "\n".join(fields)


class LRecArgs(args):

    def __init__(self, l2, loss="logistic"):
        self.l2 = l2
        self.loss = loss
