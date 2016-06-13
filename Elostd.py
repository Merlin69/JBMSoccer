import tensorflow as tf
import ToolBox
import Model as M


class Elostd(M.Model):
    def __init__(self, train_set, test_set, dictparam):
        super(Elostd, self).__init__(train_set, test_set, dictparam)

        # Define parameters
        self.elo = tf.Variable(tf.zeros([self.nb_teams, self.nb_times]))

        # Define the model
        for key, proxy in [('train', self.train_data), ('test', self.test_data)]:
            elomatch = ToolBox.get_elomatch(proxy['team_h'] - proxy['team_a'], proxy['time'], self.elo)
            elomatch += self.param['bais_ext']
            self.res[key] = tf.inv(1. + tf.exp(-elomatch))

        # Define the costs
        self.init_cost()
        for key in ['train', 'test']:
            cost = ToolBox.get_raw_elo_cost(self.param['metaparam0'], self.param['metaparam1'], self.elo, self.nb_times)
            self.regulizer[key].append(cost)

            cost = ToolBox.get_timediff_elo_cost(self.param['metaparam2'], self.elo, self.nb_times)
            self.regulizer[key].append(cost)

        # Finish the initialization
        super(Elostd, self).finish_init()

    def get_elos(self):
        return self.session.run(self.elo)