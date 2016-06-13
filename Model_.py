import tensorflow as tf
import ToolBox

ELOCONST = -1.


class Model:
    def __init__(self, train_set, test_set, dictparam, type='elostd'):
        self.type = type
        # Define constants
        self.nb_times = train_set['nb_times']
        self.nb_teams = train_set['nb_teams']
        first_time = ToolBox.first_time(self.nb_times)
        timediff = ToolBox.timediff_gen(self.nb_times)

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

        # Define parameters
        self.elo = tf.Variable(tf.zeros([self.nb_teams, self.nb_times]))

        # Define the model
        self.res = {}
        for key, proxy in [('train', self.train_data), ('test', self.test_data)]:
            elomatch = tf.matmul(proxy['team_h'] - proxy['team_a'], self.elo)
            elomatch = tf.reduce_sum(elomatch * proxy['time'], reduction_indices=[1])
            elomatch += self.param['bais_ext']
            self.res[key] = tf.inv(1. + tf.exp(ELOCONST * elomatch))

        # Define the costs
        self.cost_entropy = {}
        self.cost_regularized = {}
        for key, proxy in [('train', self.train_data), ('test', self.test_data)]:
            costs = []

            entropies = proxy['res']*tf.log(self.res[key]+1e-9) + (1-proxy['res'])*tf.log(1-self.res[key]+1e-9)
            self.cost_entropy[key] = tf.reduce_mean(-entropies)
            costs.append(self.cost_entropy[key])

            cost_rawelo = tf.reduce_mean(tf.square(tf.matmul(self.elo, first_time)))
            cost_rawelo *= self.param['metaparam1'] * ELOCONST ** 2
            cost_rawelo += tf.reduce_mean(tf.square(self.elo)) * self.param['metaparam0'] * ELOCONST ** 2
            costs.append(cost_rawelo)

            if self.nb_times > 1:
                cost_diffelo = tf.reduce_mean(tf.square(tf.matmul(self.elo, timediff)))
                cost_diffelo *= self.param['metaparam2'] * ELOCONST ** 2
                costs.append(cost_diffelo)

            self.cost_regularized[key] = tf.add_n(costs)

        # Define the cost minimization method
        self.train_step = tf.train.AdamOptimizer(0.1).minimize(self.cost_regularized['train'])

        # Create the session
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def train(self):
        self.session.run(self.train_step)

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

    def get_elos(self):
        return self.session.run(self.elo)

    def get_res(self, s):
        return self.session.run(self.res[s])

    def get_cost(self, s):
        return self.session.run(self.cost_entropy[s])

    def get_regularized_cost(self, s):
        return self.session.run(self.cost_regularized[s])

    def close(self):
        self.session.close()
