import numpy as np
import data_proc

N = data_proc.N
M = data_proc.M
train_x, train_y, _, _ = data_proc.read_mnist()

# Countint occurence
cnt = np.full((N-1, M), 0, dtype=np.int32)
for i in range(train_x.shape[0]):
    x = train_x[i].reshape((28, 28))
    y = int(train_y[i])
    for j in range(4):
        for k in range(7):
            obs = data_proc.encode_col_k_means(x.T[j*7+k])
            cnt[j*10+y][obs] += 1

total = cnt.sum(axis=1)
emission_p = np.array([[cnt[i,j] / total[i] for j in range(M)] for i in range(N-1)], dtype=float)

print('(')
for row in emission_p:
    print('    (')
    print('        %s' % (',  '.join(map(str, row))))
    print('    ),')
print(')')
