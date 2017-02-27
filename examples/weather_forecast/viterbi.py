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
A = (
        (0.80, 0.15, 0.05),
        (0.20, 0.50, 0.30),
        (0.20, 0.20, 0.60),
    )

B = (
        (0.9, 0.1),
        (0.7, 0.3),
        (0.2, 0.8),
    )

pi = (0.5, 0.25, 0.25)

hmm = DiscreteHMM(len(hidden_var), len(observation_var), A=A, B=B, pi=pi)

# Coculate the best explanation path by viterby algorithm
viterby_path = hmm.given(obs_seq)['viterby']
print('\n'.join([hidden_var_name[h_id] for h_id in viterby_path]))
