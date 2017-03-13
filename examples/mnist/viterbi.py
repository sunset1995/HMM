import os
import numpy as np
from hmm.hmm import DiscreteHMM
import data_proc

N = data_proc.N
M = data_proc.M
A = data_proc.A
B = data_proc.B
pi = data_proc.pi
hmm = DiscreteHMM(N, M, A=A, B=B, pi=pi)

# Coculate the best explanation path by viterby algorithm
_, _, test_x, test_y = data_proc.read_mnist()

correct_num = np.full((10), 0, dtype=np.int32)
total_num = np.full((10), 0, dtype=np.int32)
for i in range(test_x.shape[0]):
    x = test_x[i].reshape((28, 28))
    obs_seq = [data_proc.encode_col_k_means(x.T[i]) for i in range(28)]
    v_path = hmm.given(obs_seq)['viterby']
    
    cnt = np.full((11), 0, dtype=int)
    for hidden_state in v_path:
        if hidden_state == N-1:
            cnt[10] += 1
        else:
            cnt[hidden_state%10] += 1
    my_guess = np.argmax(cnt)

    correct_num[test_y[i]] += int(my_guess == test_y[i])
    total_num[test_y[i]] += 1

    if i and i%100 == 0:
        print('test %d datas' % i)
        print('correct rate each class:')
        print(correct_num / total_num)
        print('total correct rate:', (correct_num / total_num).mean())
        print('=====================================================')
    #print(v_path)
    #print('my_guess:', my_guess)
    #print('target  :', test_y[i])
    #print('========================================')

print('correct rate each class:')
print(correct_num / total_num)
print('total correct rate:', (correct_num / total_num).mean())

