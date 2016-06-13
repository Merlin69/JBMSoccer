import GetData
import Elostd
import Elosplit
import time
import sys

# Get the datas (50% train, 50% test)
data = GetData.Data('data.txt', 12)
data.get(0.5)

# Fix some meta-parameters
dict_param = {
    'metaparam1': 0.1,
    'metaparam0': 0.0,
    'metaparam2': 200.,
    'bais_ext': 0.3,
    'goals_bias': 0.9}

model2 = Elosplit.Elosplit(data.train, data.test, dict_param)

print model2.session.run(model2.res['train'])

for i in range(100):
    model2.train_score()
    if not i%10:
        print i, model2.get_regularized_cost_score('train'), model2.get_cost('test'), model2.get_cost('train')


sys.exit(0)
# Create the model


# Train the model
this_time = time.time()
for k in range(90):
    model.train()

# Evaluate its performance on the test set
print model.get_cost('test'), model.get_cost('train')

print int(100. * (time.time() - this_time))

# Train the model on the full dataset
data.get(1)
model.set_train_data(data.train)
for k in range(90):
    model.train()

# Get some informations on elo ratings
data.set_elos(model.get_elos())
print data.get_elos(countries=['sanmarino', 'france', 'germany', 'spain', 'romania'], times='last')
elos = data.get_elos(times='last')
elo = []
for key in elos:
    elo.append((key, elos[key][0]))
print sorted(elo, key=lambda (x, y): -y)[-5:]



# Get prediction for France vs Romania
model.set_test_data(data.create_matches([('france', 'romania', 'last')]))
print model.get_res('test')

# Close the session
model.close()
