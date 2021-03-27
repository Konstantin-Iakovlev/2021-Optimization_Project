import numpy as np
from matplotlib import pyplot as plt
import jax.numpy as jnp
from jax import jit, random
import seaborn as sns
sns.set()
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import time


class OutputOpt:
    def __init__(self, ex_time, Q_arr):
        self.ex_time = ex_time
        self.Q_arr = Q_arr

class ExperOptimizer:
    def __init__(self):
        # Generate data
        self.n = 50
        np.random.seed(1)
        digits = load_digits()
        self.X = digits["data"][:, :self.n]
        self.Y = (digits["target"] == 3).astype('int64') ### picked one class at random
        scaler = StandardScaler()
        scaler.fit_transform(self.X)

        self.M = 10
        self.L = 10
        self.N = self.X.shape[0]
        self.w1 = np.random.rand(self.n)
        self.batch_grad_size = 50
        self.batch_hess_size = 300
        # w^k
        self.w_array = [self.w1.copy()]
        # wt
        self.wt_array = []
        # learning rate
        self.beta = 2
        # EMA
        self.mu = 1e-2
        # correction pairs (sj, yj)
        self.corr_pairs = []
        # initialize real F(w1) = Q -- quality
        self.Q_arr = [self.F_maker(np.arange(0, self.N))(self.w1)]
        # precision of Q
        self.eps = 1e-5

    def f(self, w, S):
        # print(type(S), S.shape, type(self.Y), self.Y.shape)
        # print(np.exp(-self.X[S] @ w).shape)
        # return (self.Y[S] - self.X[S] @ w) ** 2
        # print('check: ', self.X[S] @ w)
        return -self.Y[S] * np.log(1/(1 + np.exp(-self.X[S] @ w))) - \
               (1-self.Y[S]) * np.log(1 - 1/(1+np.exp(-self.X[S] @ w))) +\
               0.3 * np.linalg.norm(w) ** 2

    def nabla_F(self, w, S):
        res = np.zeros(w.shape)
        for i in S:
            res += -(1 / (1 + np.exp(-self.X[i] @ w)) - self.Y[i]) * self.X[i]
        return res / self.batch_grad_size + 0.6 * w

    def hess_F(self, w, S, s_arg):
        res = np.zeros(w.shape)
        for i in S:
            # print('Xi w', self.X[i] @ w)
            aux_hess = -1/(1 + np.exp(-self.X[i] @ w))*(1 - 1/(1 + np.exp(-self.X[i] @ w))) * (self.X[i] @ s_arg) * self.X[i]
            res += aux_hess
        return res / self.batch_hess_size + 0.6 * s_arg

    def F_maker(self, S):
        def F(w):
            return (1 / S.shape[0]) * np.sum(self.f(w, S))
        return F


    def Ht(self, t):
        st, yt = self.corr_pairs[-1]
        H = (st @ yt) / (yt @ yt) * np.eye(st.shape[0])
        # print(f'corr pairs len is {len(self.corr_pairs)}')
        # print(f'min = {t, self.M, min(t, self.M)}')
        I = np.eye(st.shape[0])
        for sj, yj in self.corr_pairs[-min(t, self.M):]:
            rhoj = 1 / (yj @ sj)
            # BFGS formula
            H = (I - rhoj*np.outer(sj, yj)) @ H @ (I - rhoj*np.outer(yj, sj)) + rhoj*np.outer(sj, sj)
        # print('Ht', np.linalg.norm(H))
        return H


    def sqn_optimize(self):
        start = time.time()

        # num of iterations
        iter_num = int(5000)
        t = -1
        self.wt_array.append(np.zeros((self.w1.shape[0],)))
        # print(self.wt_array[-1])
        k = 1 # iteration counter
        while k <= iter_num:
            S = np.random.choice(self.N, size=self.batch_grad_size)
            # dF = grad(self.F_maker(S))
            self.wt_array[-1] += self.w_array[-1]
            if k <= 2 * self.L:
                # print('w_array', self.w_array[-1])
                # print(f'k = {k}')
                self.w_array.append(self.w_array[-1] - (self.beta / k) * self.nabla_F(self.w_array[-1], S))
            else:
                self.w_array.append(self.w_array[-1] -
                            (self.beta/ k) * self.Ht(t) @ self.nabla_F(self.w_array[-1], S))
                # print('w^k', self.w_array[-1])
                # print('dF(wk)', self.nabla_F(self.w_array[-1], S))
                # print(f'k = {k}')
            if k % self.L == 0:
                # print("% == 0")
                t += 1
                self.wt_array[-1] /= self.L
                if t > 0:
                    Sh = np.random.choice(self.N, size=self.batch_hess_size)
                    # hessF = hessian(self.F_maker(Sh))
                    st = self.wt_array[-1] - self.wt_array[-2]
                    yt = self.hess_F(self.wt_array[-1], Sh, st)
                    # print('wt = ', self.wt_array[-1])
                    # print('yt = ', yt)
                    self.corr_pairs.append((st, yt))
                self.wt_array.append(np.zeros((self.w1.shape[0],)))

            self.Q_arr.append(self.mu*self.F_maker(S)(self.w_array[-1]) + (1-self.mu)*self.Q_arr[-1])
            if abs(self.Q_arr[-1] - self.Q_arr[-2]) < self.eps:
                print(f'sqn iterations = {k}')
                break
            k += 1
        finish = time.time()
        res = OutputOpt(finish - start, self.Q_arr)
        self.clear()
        return res

    def sgd_optimize(self):
        start = time.time()
        iter_num = int(5000)
        k = 1
        while k <= iter_num:
            S = random.choice(random.PRNGKey(np.random.randint(1)),self.N, shape=(self.batch_grad_size,))
            self.w_array.append(self.w_array[-1] - 0.01*self.nabla_F(self.w_array[-1], S))
            self.Q_arr.append(self.mu*self.F_maker(S)(self.w_array[-1]) + (1-self.mu)*self.Q_arr[-1])
            if abs(self.Q_arr[-1] - self.Q_arr[-2]) < self.eps:
                self.processed_iter = k
                self.opt_Q = self.Q_arr[-1]
                print(f'sgd_opt iter = {k}')
                break
            k += 1
        finish = time.time()
        res = OutputOpt(finish - start, self.Q_arr)
        self.clear()
        return res

    def clear(self):
        self.Q_arr = [self.F_maker(np.arange(0, self.N))(self.w1)]
        self.wt_array = []
        self.w_array = [self.w1]


# TEST
np.random.seed(1)
sqn_opt = ExperOptimizer()
sqn_res = sqn_opt.sqn_optimize()
print("Ok")

sgd_res = sqn_opt.sgd_optimize()
print('Ok')


plt.figure(figsize=(10, 8))
plt.plot(np.log(sqn_res.Q_arr), label=f'SQN, time = {sqn_res.ex_time:.3f} sec')
plt.plot(np.log(sgd_res.Q_arr), label=f'SGD, time = {sgd_res.ex_time:.3f} sec')
plt.legend()
plt.xlabel('number of iteration')
plt.ylabel('$\log Q$')
plt.show()