import tensorflow as tf
import ToolBox
import Model as M


class YOUR_CLASS_NAME(M.Model):
    def __init__(self, train_set, test_set, dictparam):
        super(YOUR_CLASS_NAME, self).__init__(train_set, test_set, dictparam)

        # Define parameters

        # Define the model
        for key, proxy in [('train', self.train_data), ('test', self.test_data)]:
            self.res[key] = None

        # Define the costs
        self.init_cost()
        for key, proxy in [('train', self.train_data), ('test', self.test_data)]:
            self.costs_res[key].append(None)

        # Finish the initialization
        super(YOUR_CLASS_NAME, self).finish_init()
