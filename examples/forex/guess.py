import os
import numpy as np
from hmm.hmm import DiscreteHMM

# Mapping between raw observed datas and features
EPS = 0.00001
def obs_2_id(obs):
    if obs > 3*EPS:
        return 4
    elif obs > EPS:
        return 3
    elif obs < -3*EPS:
        return 2
    elif obs < -EPS:
        return 1
    else:
        return 0

id_2_obs = {
    0: '=',
    1: '<',
    2: '<<',
    3: '>',
    4: '>>',
}

# Reading input file
raw_seq = []
with open(os.path.join(os.path.dirname(__file__), 'input.txt')) as f:
    for line in f:
        raw_seq.append(float(line.strip()))

obs_seq = [0, ] * (len(raw_seq)-1)
for t in range(len(raw_seq)-1):
    obs_seq[t] = obs_2_id(raw_seq[t+1] - raw_seq[t])

# Setting model
num_hidden_var = 6
num_obs_var = len(id_2_obs)
num_train_seq = 10

# Training the model best describe the observation
good_correct = 0
good_total = 0
correct_num = 0
total = 0
for t in range(num_train_seq, len(obs_seq)-1):
    hmm = DiscreteHMM(num_hidden_var, num_obs_var)
    hmm.train(obs_seq[t-num_train_seq : t])
    p = np.array(hmm.given(obs_seq[t-num_train_seq : t])['forward'])
    belief = hmm.B.T.dot(hmm.A.T.dot(p))
    guess = np.argmax(belief)

    np.set_printoptions(precision=2, suppress=True)
    print('belief:', belief)
    print('guess : %7s' % id_2_obs[guess])
    print('target: %7s' % id_2_obs[obs_seq[t]])

    if guess == obs_seq[t]:
        print('result: correct')
        correct_num += 1
    else:
        print('result:   wrong')
    total += 1
    print('correct rate so far: %.2f' % (correct_num / total))

    if id_2_obs[guess] in ('>', '>>') and obs_seq[t] != 0:
        good_correct += int(id_2_obs[obs_seq[t]] in ('>', '>>'))
        good_total += 1
    if good_total:
        print('good guess rate so far: %.2f' % (good_correct / good_total))
    print('==================================================================')
