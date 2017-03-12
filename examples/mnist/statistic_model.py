import numpy as np
import data_proc

train_x, train_y, test_x, test_y = data_proc.read_mnist()

# Countint occurence
cnt = np.full((40, 64), 0, dtype=np.int32)
for i in range(train_x.shape[0]):
    x = train_x[i].reshape((28, 28))
    y = int(train_y[i])
    for j in range(4):
        for k in range(7):
            obs = data_proc.encode_col(x.T[j*7+k])
            cnt[j*10+y][obs] += 1

total = cnt.sum(axis=1)
emission_p = np.array([[cnt[i,j] / total[i] for j in range(64)] for i in range(40)], dtype=float)

print('(')
for row in emission_p:
    print('    (')
    print('        %s' % (',  '.join(map(str, row))))
    print('    ),')
print(')')
