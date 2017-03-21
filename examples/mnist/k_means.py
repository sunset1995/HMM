import sys
import random
import numpy as np
import data_proc

if len(sys.argv) != 2:
    print('Please specify output file name')
    exit(0)

try:
    with open(sys.argv[1], 'w') as f:
        pass
except:
    print('Error while opening the output file')
    sys.exit(0)

train_x, _, _, _ = data_proc.read_mnist()

# Get all points
points = set()
for i in range(train_x.shape[0]):
    x = train_x[i].reshape((28, 28))
    for j in range(1, 28):
        points.add(tuple(x.T[j-1:j+1].flat))

points.remove((0, )*56)
points = [np.array(p) for p in points]
print(len(points))

# K-means
# Special case: always make (0, ) * 28 be one of the center
M = data_proc.M
centers = []
members_sum = []
members_num = []
converge = False
def init_centers():
    global centers
    centers = []
    for i in range(M-1):
        centers.append(random.choice(points))

def update_member():
    global members_sum
    global members_num
    global converge
    last_num = members_num
    members_sum = np.full((M-1, 56), 0, dtype=np.int64)
    members_num = np.full((M-1), 0, dtype=np.int64)
    for p in points:
        closest = np.argmin(((p-centers)**2).sum(axis=1))
        members_sum[closest] += p
        members_num[closest] += 1
    if np.array_equal(last_num, members_num):
        converge = True

def update_center():
    global centers
    for i in range(M-1):
        centers[i] = members_sum[i] / members_num[i]


init_centers()
for i in range(1000):
    update_member()
    update_center()
    
    # Print temporary result at each iteration
    with open(sys.argv[1], 'w') as f:
        print('iteration %d' % (i), file=f)
        for c in centers:
            print(tuple(c), file=f)
        print((0, ) * 56, file=f)
        print(list(members_num), file=f)
        if converge:
            print('converge', file=f)
            break
