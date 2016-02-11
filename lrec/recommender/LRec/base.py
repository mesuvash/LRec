import scipy.sparse


class BaseLinear(object):
    """docstring for BaseLinear"""

    def __init__(self):
        super(BaseLinear, self).__init__()

    def get_sim(self):
        return self.sim

    def recommend_all(self, train_input):
        return train_input * self.sim

    def recommend(self, user, train_input):
        result = train_input[user, :] * self.sim
        if scipy.sparse.issparse(result):
            result = result.todense()
        return result
