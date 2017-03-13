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

v_correct_num = np.full((10), 0, dtype=np.int32)
v_total_num = np.full((10), 0, dtype=np.int32)
p_correct_num = np.full((10), 0, dtype=np.int32)
p_total_num = np.full((10), 0, dtype=np.int32)
for i in range(test_x.shape[0]):
    x = test_x[i].reshape((28, 28))
    obs_seq = [data_proc.encode_col_k_means(x.T[i]) for i in range(28)]

    path = [hmm.given(obs_seq[0:1])]
    for j in range(1, 28):
        path.append(hmm.given_more(obs_seq[j:j+1]))
    v_path = path[-1]['viterby']
    p_path = [np.argmax(belief['forward']) for belief in path]
    #v_path = hmm.given(obs_seq)['viterby']
    
    # Viterbi guessing
    v_cnt = np.full((11), 0, dtype=int)
    for hidden_state in v_path:
        if hidden_state != N-1:
            v_cnt[hidden_state%10] += 1
    v_guess = np.argmax(v_cnt)

    # Forward guessing
    p_cnt = np.full((11), 0, dtype=int)
    for hidden_state in p_path:
        if hidden_state != N-1:
            p_cnt[hidden_state%10] += 1
    p_guess = np.argmax(p_cnt)

    # Statistic result
    v_correct_num[test_y[i]] += int(v_guess == test_y[i])
    v_total_num[test_y[i]] += 1
    p_correct_num[test_y[i]] += int(p_guess == test_y[i])
    p_total_num[test_y[i]] += 1

    if i and i%100 == 0:
        print('test %d datas' % i)
        print('viterby correct rate each class:')
        print(v_correct_num / v_total_num)
        print('forward correct rate each class:')
        print(p_correct_num / p_total_num)
        print('viterby total correct rate:', (v_correct_num / v_total_num).mean())
        print('forward total correct rate:', (p_correct_num / p_total_num).mean())
        print('=====================================================')
    
    #print(p_path)
    #print(v_path)
    #print('viterby guess:', v_guess)
    #print('forward guess:', p_guess)
    #print('target       :', test_y[i])
    #print('========================================')

print('correct rate each class:')
print(correct_num / total_num)
print('total correct rate:', (correct_num / total_num).mean())

