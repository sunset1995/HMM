import random
import numpy as np
import data_proc

train_x, _, _, _ = data_proc.read_mnist()

# Get all points
points = [np.full((28), 0, dtype=np.uint8)]
for i in range(train_x.shape[0]):
    x = train_x[i].reshape((28, 28))
    for j in range(28):
        if not np.array_equal(x.T[j], points[0]):
            points.append(x.T[j])

print(len(points))

# K-means
M = 20
centers = []
members_sum = []
members_num = []
def init_centers():
    global centers
    centers = [points[0]]
    for i in range(M-1):
        centers.append(random.choice(points))

def update_member():
    global members_sum
    global members_num
    members_sum = np.full((M, 28), 0, dtype=np.int64)
    members_num = np.full((M), 0, dtype=np.int64)
    for p in points:
        closest = np.argmin(((p-centers)**2).sum(axis=1))
        members_sum[closest] += p
        members_num[closest] += 1

def update_center():
    global centers
    for i in range(M):
        centers[i] = members_sum[i] / members_num[i]


init_centers()
for i in range(100):
    update_member()
    update_center()
    for c in centers:
        print(tuple(c))
    print(list(members_num))
    print('=============================================')
