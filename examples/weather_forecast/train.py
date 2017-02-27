import os
from hmm.hmm import DiscreteHMM

# Mapping input to variable's id
hidden_var_name = ('sunny', 'foggy', 'rainy')
observation_var_name = ('no', 'yes')
hidden_var = {
    hidden_var_name[0]: 0,
    hidden_var_name[1]: 1,
    hidden_var_name[2]: 2,
    }

observation_var = {
    observation_var_name[0]: 0,
    observation_var_name[1]: 1,
    }

target = []
obs_seq = []

with open(os.path.join(os.path.dirname(__file__), 'input.txt')) as f:
    for line in f:
        hidden, observe = line.strip().split(',')
        target.append(hidden_var[hidden])
        obs_seq.append(observation_var[observe])

# Setting model
B = (
        (0.8, 0.2),
        (0.5, 0.5),
        (0.1, 0.9),
    )
hmm = DiscreteHMM(len(hidden_var), len(observation_var), B=B)

# Training the model best describe the observation
hmm.train(obs_seq, verbose=1)
hmm.show_model()
print('checksum:', hmm.check_model())
