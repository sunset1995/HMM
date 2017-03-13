import math
import numpy as np
from . import util

class DiscreteHMM:
    def __init__(self, N, M, A=None, B=None, pi=None):
        self.N = N
        self.M = M
        self.__obs_seq = None
        self.__viterby = None
        self.__log_p_vit = None
        self.__log_p_fwd = None

        if A is not None:
            self.A = np.array(A, dtype=np.float64)
        else:
            self.A = util.normalize2d(np.random.rand(self.N, self.N))

        if B is not None:
            self.B = np.array(B, dtype=np.float64)
        else:
            self.B = util.normalize2d(np.random.rand(self.N, self.M))

        if pi is not None:
            self.pi = np.array(pi, dtype=np.float64)
        else:
            self.pi = util.normalize1d(np.random.rand(self.N))

        # Remove zero
        self.A += ((self.A < util.ZERO) * util.ZERO).astype(np.float64)
        self.B += ((self.B < util.ZERO) * util.ZERO).astype(np.float64)
        self.pi += ((self.pi < util.ZERO) * util.ZERO).astype(np.float64)

        self.log_A = np.log(self.A)
        self.log_B = np.log(self.B)
        self.log_pi = np.log(self.pi)

        assert(self.A.shape == (N, N, ))
        assert(self.B.shape == (N, M, ))
        assert(self.pi.shape == (N, ))
        assert(self.check_model())

    def __check_obs_seq(self, obs_seq):
        for obs in obs_seq:
            assert(obs >= 0 and obs < self.M)

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

    def __forward_backward(self, alpha, beta):
        gamma = np.full((len(alpha), self.N), util.LOG_ZERO, dtype=np.float64)
        for t in range(len(alpha)):
            gamma[t] = util.log_vec_mul(alpha[t], beta[t])
            gamma[t] = util.log_vec_div(gamma[t], util.log_sum(*gamma[t]))
        return gamma

    def __xi_t(self, t, alpha, beta, obs):
        xi_t = np.full((self.N, self.N), util.LOG_ZERO, dtype=np.float64)
        for i in range(self.N):
            xi_t[i] = util.log_vec_mul(util.log_vec_mul(self.log_A[i, :], self.log_B[:, obs[t+1]], beta[t+1, :]), alpha[t][i])
        xi_t = util.log_vec_div(xi_t, util.log_sum(*xi_t.reshape(xi_t.size)))
        return xi_t

    def __optimize_model(self, obs):
        alpha = self.__forward(obs)
        beta = self.__backward(obs)
        gamma = self.__forward_backward(alpha, beta)

        A_son = np.full((self.N, self.N), util.LOG_ZERO, dtype=np.float64)
        A_mom = np.full((self.N), util.LOG_ZERO, dtype=np.float64)
        B_son = np.full((self.N, self.M), util.LOG_ZERO, dtype=np.float64)
        B_mom = np.full((self.N), util.LOG_ZERO, dtype=np.float64)
        n_pi = np.full((self.N), util.LOG_ZERO, dtype=np.float64)

        # Count transition probability
        for t in range(len(obs)-1):
            xi_t = self.__xi_t(t, alpha, beta, obs)
            A_son = util.log_vec_add(A_son, xi_t)
            A_mom = util.log_vec_add(A_mom, gamma[t])

        # Count emmision probability
        for t in range(len(obs)):
            B_son[:, obs[t]] = util.log_vec_add(B_son[:, obs[t]], gamma[t])
            B_mom = util.log_vec_add(B_mom, gamma[t])

        # Assign new better model
        self.log_A = util.log_vec_div(A_son.T, A_mom).T
        self.log_B = util.log_vec_div(B_son.T, B_mom).T
        self.log_pi = gamma[0]
        A = np.exp(self.log_A)
        B = np.exp(self.log_B)
        pi = np.exp(self.log_pi)

        delta = np.sum(np.fabs(np.array((
            *(A-self.A).flatten(),
            *(B-self.B).flatten(),
            *(pi-self.pi)))))

        self.A = A
        self.B = B
        self.pi = pi

        return delta


    def show_model(self):
        np.set_printoptions(precision=4, suppress=True)
        print('A: Transition probability'.center(70, '-'))
        print(self.A)
        print('B: Emission probability'.center(70, '-'))
        print(self.B)
        print('pi: initital state distribution'.center(70, '-'))
        print(self.pi)

    def check_model(self):
        return abs(self.A.sum() - self.N) < util.EPS \
            and abs(self.B.sum() - self.N) < util.EPS \
            and abs(self.pi.sum() - 1.0) < util.EPS

    def given(self, obs_seq):
        self.__check_obs_seq(obs_seq)
        self.__obs_seq = []
        self.__viterby = []
        self.__log_p_vit = np.array(self.log_pi)
        self.__log_p_fwd = np.array(self.log_pi)
        return self.given_more(obs_seq)

    def given_more(self, obs_seq):
        self.__check_obs_seq(obs_seq)
        if self.__obs_seq is None:
            return self.given(obs_seq)
        
        for obs in obs_seq:
            # Coculate probability distribution via forward algorithm
            if len(self.__obs_seq) == 0:
                self.__log_p_fwd = util.log_mul(self.__log_p_fwd, self.log_B[:, obs])
            else:
                self.__log_p_fwd = np.array([
                    util.log_mul(util.log_sum(*util.log_vec_mul(self.__log_p_fwd, self.log_A[:, i])), self.log_B[i, obs])
                        for i in range(self.N)])
            # Normalize
            self.__log_p_fwd = util.log_vec_div(self.__log_p_fwd, util.log_sum(*self.__log_p_fwd))

            # Coculate best explanation via viterby algorithm
            if len(self.__obs_seq) == 0:
                self.__viterby = list([i] for i in range(self.N))
                self.__log_p_vit = util.log_mul(self.__log_p_vit, self.log_B[:, obs])
            else:
                best_prev = [np.argmax(util.log_vec_mul(self.__log_p_vit, self.log_A[:, i])) for i in range(self.N)]
                self.__viterby = [[*self.__viterby[best_prev[i]], i] for i in range(self.N)]
                self.__log_p_vit = np.array([
                    util.log_mul(self.__log_p_vit[best_prev[i]], self.log_A[best_prev[i], i], self.log_B[i, obs])
                        for i in range(self.N)])

            self.__obs_seq.append(obs)

        return {
            'forward': np.exp(self.__log_p_fwd),
            'viterby': list(self.__viterby[np.argmax(self.__log_p_vit)]),
        }

    def train(self, obs_seq, itnum=100, eps=0.01, verbose=0):
        self.__check_obs_seq(obs_seq)
        for _ in range(itnum):
            delta = self.__optimize_model(obs_seq)
            
            if verbose > 0:
                print('itnum %5d : delta %f' % (_+1, delta))
            if verbose > 1:
                self.show_model()

            if abs(delta) < eps:
                break
