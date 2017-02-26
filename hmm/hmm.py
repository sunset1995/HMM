import math
import numpy as np
from . import util

class DiscreteHMM:
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.__random_init_model()

    def __random_init_model(self):
        # Random assign value
        self.log_A = np.log(util.normalize2d(np.random.rand(self.N, self.N)))
        self.log_B = np.log(util.normalize2d(np.random.rand(self.N, self.M)))
        self.log_pi = np.log(util.normalize1d(np.random.rand(self.N)))

    def __forward(self, obs):
        # Return alpha via forward algorithm
        # All coculations and results are in log space
        alpha = np.full((len(obs), self.N), util.LOG_ZERO, dtype=np.float64)
        alpha[0] = util.log_mul(self.log_pi, self.log_B[:, obs[0]])
        for t in range(1, len(obs)):
            for j in range(self.N):
                alpha[t][j] = util.log_mul(util.log_sum(*util.log_vec_mul(self.log_A[:, j], alpha[t-1])), self.log_B[j, obs[t]])
        return alpha

    def __backward(self, obs):
        beta = np.full((len(obs), self.N), math.log(1.0), dtype=np.float64)
        for t in range(len(obs)-2, -1, -1):
            for i in range(self.N):
                beta[t][i] = util.log_sum(*util.log_vec_mul(self.log_A[i, :], self.log_B[:, obs[t+1]], beta[t+1]))
        return beta

    def show_model(self):
        print('A: Transition probability'.center(70, '-'))
        print(np.exp(self.log_A))
        print('B: Emission probability'.center(70, '-'))
        print(np.exp(self.log_B))
        print('pi: initital state distribution'.center(70, '-'))
        print(np.exp(self.log_pi))

    def check_model(self):
        return abs(np.sum(np.exp(self.log_A)) - self.N) < util.EPS \
            and abs(np.sum(np.exp(self.log_B)) - self.N) < util.EPS \
            and abs(np.sum(np.exp(self.log_pi)) - 1.0) < util.EPS

    def train(self, obs):
        pass
