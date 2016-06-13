import tensorflow as tf
import copy

class Model(object):
    def __init__(self, train_set, test_set, dictparam):
        # Define constants
        self.nb_times = train_set['nb_times']
        self.nb_teams = train_set['nb_teams']

        # Define meta-parameters
        self.param = {}
        for key in dictparam:
            self.param[key] = tf.Variable(dictparam[key], trainable=False)

        # Define training and testing set
        self.train_data = {}
        self.test_data = {}
        for key in train_set:
            self.train_data[key] = tf.Variable(train_set[key], validate_shape=False, trainable=False)
        for key in test_set:
            self.test_data[key] = tf.Variable(test_set[key], validate_shape=False, trainable=False)

        # Define child variables
        self.regulizer = {}
        self.cost_entropy_res = {}
        self.cost_regularized_res = {}
        self.cost_entropy_score = {}
        self.cost_regularized_score = {}
        self.train_step_res = None
        self.train_step_score = None
        self.session = None
        self.res = {}
        self.score = {}

        self.pr = None
        self.entropies_score = None

    def init_cost(self):
        for key, proxy in [('train', self.train_data), ('test', self.test_data)]:
            entropies_res = proxy['res'] * tf.log(self.res[key] + 1e-9) + (1 - proxy['res']) * tf.log(
                1 - self.res[key] + 1e-9)
            self.cost_entropy_res[key] = tf.reduce_mean(-entropies_res)
            if key in self.score:
                self.pr = tf.squeeze(tf.batch_matmul(tf.expand_dims(proxy['score_h'], 1), self.score[key]), [1])
                self.entropies_score = tf.log(tf.reduce_sum(self.pr * proxy['score_a'], reduction_indices=[1]) + 1e-9)
                self.cost_entropy_score[key] = tf.reduce_mean(-self.entropies_score)

            self.regulizer[key] = []

    def finish_init(self):
        for key in ['train', 'test']:
            costs_res = copy.copy(self.regulizer[key])
            costs_res.append(self.cost_entropy_res[key])
            self.cost_regularized_res[key] = tf.add_n(costs_res)

            if key in self.cost_entropy_score:
                costs_score = copy.copy(self.regulizer[key])
                costs_score.append(self.cost_entropy_score[key])
                self.cost_regularized_score[key] = tf.add_n(costs_score)

        # Define the cost minimization method
        self.train_step_res = tf.train.AdamOptimizer(0.1).minimize(self.cost_regularized_res['train'])
        if 'train' in self.cost_regularized_score:
            self.train_step_score = tf.train.AdamOptimizer(0.1).minimize(self.cost_regularized_score['train'])

        # Create the session
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def train_res(self):
        self.session.run(self.train_step_res)

    def train_score(self):
        self.session.run(self.train_step_score)

    def set_params(self, dictparam):
        for key in dictparam:
            if key in self.param:
                self.session.run(tf.assign(self.param[key], dictparam[key]))
            else:
                raise ValueError('invalid dictparam key in set_params')

    def set_train_data(self, dataset):
        for key in dataset:
            if key in self.train_data:
                self.session.run(tf.assign(self.train_data[key], dataset[key], validate_shape=False))
            else:
                raise ValueError('invalid dataset key in set_datas')

    def set_test_data(self, dataset):
        for key in dataset:
            if key in self.train_data:
                self.session.run(tf.assign(self.test_data[key], dataset[key], validate_shape=False))
            else:
                raise ValueError('invalid dataset key in set_datas')

    def get_res(self, s):
        return self.session.run(self.res[s])

    def get_cost(self, s):
        return self.session.run(self.cost_entropy_res[s])

    def get_regularized_cost(self, s):
        return self.session.run(self.cost_regularized_res[s])

    def get_cost_score(self, s):
        return self.session.run(self.cost_entropy_score[s])

    def get_regularized_cost_score(self, s):
        return self.session.run(self.cost_regularized_score[s])

    def close(self):
        self.session.close()
