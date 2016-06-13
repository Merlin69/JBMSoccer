import GetData
import Model_
import time

data = GetData.Data('data.txt', 12)



# Fix some meta-parameters
dict_param = {
    'metaparam1': 0.01,
    'metaparam0': 0.0,
    'metaparam2': 3.6,
    'bais_ext': 0.3}

# Create the model
model = Model_.Model(data.train, data.test, dict_param)

# Train the model
full_time = time.time()
this_time = time.time()
for i in range(10):
    for k in range(90):
        model.train()
    print "train", 100 * (time.time() - this_time)
    this_time = time.time()

    for key in model.param:
        print key, model.session.run(model.param[key])
    print "print params", 100 * (time.time() - this_time)
    this_time = time.time()

    model.update_params()
    print "update params", 100 * (time.time() - this_time)
    this_time = time.time()

    for key in model.param:
        print key, model.session.run(model.param[key])
    print "print params", 100 * (time.time() - this_time)
    this_time = time.time()

    dict_parambis = {
        'metaparam1': 0.02,
        'metaparam0': 0.0,
        'metaparam2': 4.,
        'bais_ext': 0.2}
    model.set_params(dict_parambis)
    print "set params", 100*(time.time() - this_time)
    this_time = time.time()

    u = model.get_cost('test')
    print "get cost", 100*(time.time() - this_time)
    this_time = time.time()

print "full time", 10*(time.time() - full_time)
