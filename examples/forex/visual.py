import os
import matplotlib.pyplot as plt

# Reading input file
raw_seq = []
with open(os.path.join(os.path.dirname(__file__), 'input.txt')) as f:
    for line in f:
        raw_seq.append(float(line.strip()))

obs_seq = [0, ] * (len(raw_seq)-1)
for t in range(len(raw_seq)-1):
    obs_seq[t] = raw_seq[t+1] - raw_seq[t]

plt.plot(raw_seq)
plt.show()
