import tensorflow as tf
import ToolBox
import Model as M


class Elosplit(M.Model):
    def __init__(self, train_set, test_set, dictparam):
        super(Elosplit, self).__init__(train_set, test_set, dictparam)

        k = tf.constant(map(lambda x: float(x), range(10)))
        last_vect = tf.expand_dims(ToolBox.last_vector(10),0)
        win_vector = ToolBox.win_vector(10)

        # Define parameters
        self.elo_atk = tf.Variable(tf.zeros([self.nb_teams, self.nb_times]))
        self.elo_def = tf.Variable(tf.zeros([self.nb_teams, self.nb_times]))

        # Define the model
        for key, proxy in [('train', self.train_data), ('test', self.test_data)]:
            elo_atk_h = ToolBox.get_elomatch(proxy['team_h'], proxy['time'], self.elo_atk)
            elo_def_h = ToolBox.get_elomatch(proxy['team_h'], proxy['time'], self.elo_def)
            elo_atk_a = ToolBox.get_elomatch(proxy['team_a'], proxy['time'], self.elo_atk)
            elo_def_a = ToolBox.get_elomatch(proxy['team_a'], proxy['time'], self.elo_def)
            lambda_h = tf.expand_dims(tf.exp(self.param['goals_bias'] + elo_atk_h - elo_def_a), 1)
            lambda_a = tf.expand_dims(tf.exp(self.param['goals_bias'] + elo_atk_a - elo_def_h), 1)
            score_h = tf.exp(-lambda_h + tf.log(lambda_h) * k - tf.lgamma(k + 1))
            score_a = tf.exp(-lambda_a + tf.log(lambda_a) * k - tf.lgamma(k + 1))
            score_h += tf.matmul(tf.expand_dims((1. - tf.reduce_sum(score_h, reduction_indices=[1])), 1), last_vect)
            score_a += tf.matmul(tf.expand_dims((1. - tf.reduce_sum(score_a, reduction_indices=[1])), 1), last_vect)

            self.score[key] = tf.batch_matmul(tf.expand_dims(score_h, 2), tf.expand_dims(score_a, 1))
            self.res[key] = tf.reduce_sum(self.score[key] * win_vector, reduction_indices=[1,2])

        # Define the costs
        self.init_cost()
        for key in ['train', 'test']:
            for proxy in [self.elo_atk, self.elo_def]:
                cost = ToolBox.get_raw_elo_cost(self.param['metaparam0'], self.param['metaparam1'], proxy, self.nb_times)
                self.regulizer[key].append(cost)

                cost = ToolBox.get_timediff_elo_cost(self.param['metaparam2'], proxy, self.nb_times)
                self.regulizer[key].append(cost)

        # Finish the initialization
        super(Elosplit, self).finish_init()

    def get_elos(self):
        return self.session.run(self.elo_atk)