# coding=utf-8
import numpy as np
from scipy.stats import norm
from .utils.utils import update_dumping
from .utils.utils import poisson_average


class StabilitySelectionVAMPSolver(object):
    """ naive Self averaging VAMP solver for stability selection.
    this version is a naive implementation in the sense that
    inverse calculations of N x N and M x M matrices are used.
    """

    def __init__(self, A, y, regularization_strength, dumping_coefficient, mu, enlarge_prob=0.5, enlarge_ratio=0.5,
                 clip_min=1e-10, clip_max=1e10):
        """constructor

        Args:
            A: observation matrix
            y: observed values
            regularization_strength: coefficient of the l1 regularization
            dumping_coefficient: dumping coefficient
            mu: mean of poisson dist
            enlarge_prob: with probability enlarge_prob, regularization strength is enlarged
            enlarge_ratio: regularization strength is enlarged as ¥lambda -> (¥lambda / enlarge_ratio)
        """
        self.A = A.copy()
        self.AT = self.A.T
        self.y = y.copy()
        self.M, self.N = A.shape

        self.J1 = self.A.T @ self.A
        self.J2 = self.A @ self.A.T

        self.l = regularization_strength * np.ones(self.N)
        self.eta = dumping_coefficient
        self.mu = mu

        self.enlarge_prob = enlarge_prob
        self.enlarge_ratio = enlarge_ratio

        self.clip_min = clip_min
        self.clip_max = clip_max

        ''' message from 2 to 1 '''
        # x
        self.r1x = np.zeros(self.N)
        self.chi1x_hat = np.ones(self.N)
        self.q1x_hat = np.ones(self.N)

        # u
        self.r1u = np.zeros(self.M)
        self.chi1u_hat = np.ones(self.M)
        self.q1u_hat = np.ones(self.M)

        ''' variable 1 estimation '''
        # x
        self.x1_hat = np.random.normal(0.0, 1.0, self.N)
        self.chi1x = np.ones(self.N)
        self.v1x = np.ones(self.N)
        self.eta1x_1 = np.ones(self.N)
        self.eta1x_2 = np.ones(self.N)
        # u
        self.u1_hat = np.random.normal(0.0, 1.0, self.M)
        self.chi1u = np.ones(self.M)
        self.v1u = np.ones(self.M)
        self.eta1u_1 = np.ones(self.M)
        self.eta1u_2 = np.ones(self.M)

        ''' message from 1 to 2 '''
        # x
        self.r2x = np.random.normal(0.0, 1.0, self.N)
        self.chi2x_hat = np.ones(self.N)
        self.q2x_hat = np.ones(self.N)
        # u
        self.r2u = np.random.normal(0.0, 1.0, self.M)
        self.chi2u_hat = np.ones(self.M)
        self.q2u_hat = np.ones(self.M)

        ''' variable 2 estimation '''
        # x
        self.x2_hat = np.random.normal(0.0, 1.0, self.N)
        self.chi2x = np.ones(self.N)
        self.v2x = np.ones(self.N)
        self.eta2x_1 = np.ones(self.N)
        self.eta2x_2 = np.ones(self.N)
        # u
        self.u2_hat = np.random.normal(0.0, 1.0, self.M)
        self.chi2u = np.ones(self.M)
        self.v2u = np.ones(self.M)
        self.eta2u_1 = np.ones(self.M)
        self.eta2u_2 = np.ones(self.M)

        self.variable_history = []
        self.start_history = []
        self.diff_history = []

        self.iteration_index = None

    def solve(self, max_iteration, tolerance, message):
        """VAMP Solver for stability selection

        Args:
            max_iteration: max number of iterations
            tolerance: stopping criterion
            message: convergence info

        Returns:
            None
        """
        convergence_flag = False
        abs_diff = 99999
        abs_diff_v = 99999
        iteration_index = -1
        for iteration_index in range(max_iteration):
            # print(iteration_index)
            self.variable_history.append(self.return_variables())
            if iteration_index == 0:
                self.start_history.append(1)
            else:
                self.start_history.append(0)

            self.variable1_estimation()
            self.calculate_message_from1to2()
            self.variable2_estimation()
            self.calculate_message_from2to1()

            abs_diff = np.linalg.norm(self.x1_hat - self.x2_hat) / np.sqrt(self.N)
            abs_diff_v = np.linalg.norm(self.v1x - self.v2x) / np.sqrt(self.N)

            diff = max(abs_diff, abs_diff_v)
            self.diff_history.append(diff)
            if diff < tolerance and iteration_index > 1:
                convergence_flag = True
                break

        self.iteration_index = iteration_index + 1
        if convergence_flag:
            if message:
                print("converged")
                print("abs_diff_x= {0} ".format(abs_diff))
                print("abs_diff_v= {0} ".format(abs_diff_v))
                print("diff between 1 and 2, x hat = {0}".format(
                    np.linalg.norm(self.x1_hat - self.x2_hat) / np.sqrt(self.N)))
                print("diff between 1 and 2, x v = {0}".format(np.linalg.norm(self.v1x - self.v2x) / np.sqrt(self.N)))

                print("iteration num={0}".format(iteration_index))
                # self.show_me()
                print()

        else:
            print("doesn't converged")
            print("abs_diff= {0} ".format(abs_diff))
            print("abs_diff_v= {0} ".format(abs_diff_v))
            print("iteration num={0}".format(iteration_index))
            self.show_me()
            print()
        return convergence_flag

    # variable 1 estimation
    def variable1_estimation(self):
        """ estimate variables at variable node 1 """
        # x
        self.x1_hat = update_dumping(old_x=self.x1_hat,
                                     new_x=self._calculate_x1_hat(),
                                     dumping_coefficient=self.eta)
        new_chi1x, new_v1x = self._calculate_chi1x(), self._calculate_v1x()
        self.chi1x = update_dumping(old_x=self.chi1x,
                                    new_x=np.clip(a=new_chi1x, a_min=self.clip_min, a_max=self.clip_max),
                                    dumping_coefficient=self.eta)
        self.v1x = update_dumping(old_x=self.v1x,
                                  new_x=np.clip(a=new_v1x, a_min=self.clip_min, a_max=self.clip_max),
                                  dumping_coefficient=self.eta)
        self.eta1x_1, self.eta1x_2 = 1.0 / self.chi1x, self.v1x / (self.chi1x ** 2.0)

        # u
        self.u1_hat = update_dumping(old_x=self.u1_hat,
                                     new_x=self._calculate_u1_hat(),
                                     dumping_coefficient=self.eta)
        new_chi1u, new_v1u = self._calculate_chi1u(), self._calculate_v1u()
        self.chi1u = update_dumping(old_x=self.chi1u,
                                    new_x=np.clip(a=new_chi1u, a_min=self.clip_min, a_max=self.clip_max),
                                    dumping_coefficient=self.eta)
        self.v1u = update_dumping(old_x=self.v1u,
                                  new_x=np.clip(a=new_v1u, a_min=self.clip_min, a_max=self.clip_max),
                                  dumping_coefficient=self.eta)
        self.eta1u_1, self.eta1u_2 = 1.0 / self.chi1u, self.v1u / (self.chi1u ** 2.0)

    def _calculate_x1_hat(self):
        """ calculate x1 hat """

        def get_new_x1_hat(chi1x_hat, h1x, l, q1x_hat):
            z_minus = -1.0 * (h1x + l) / np.sqrt(chi1x_hat)
            z_plus = -1.0 * (h1x - l) / np.sqrt(chi1x_hat)
            first_term = (h1x + l) * norm.cdf(z_minus)
            second_term = (h1x - l) * norm.sf(z_plus)
            third_term = np.sqrt(chi1x_hat / 2.0 / np.pi) * (
                    -np.exp(-z_minus ** 2.0 / 2.0) + np.exp(-z_plus ** 2.0 / 2.0)
            )
            new_x1_hat = (first_term + second_term + third_term) / q1x_hat
            return new_x1_hat

        h1x = self.q1x_hat * self.r1x
        l = self.l
        chi1x_hat = self.chi1x_hat
        q1x_hat = self.q1x_hat

        new_x1_hat = self.enlarge_prob * get_new_x1_hat(chi1x_hat, h1x, l / self.enlarge_ratio, q1x_hat) + (
                1.0 - self.enlarge_prob) * get_new_x1_hat(chi1x_hat, h1x, l, q1x_hat)
        return new_x1_hat

    def _calculate_chi1x(self):
        """ calculate chi1x """

        def get_new_chi1x(chi1x_hat, h1x, l, q1x_hat):
            z_minus = -1.0 * (h1x + l) / np.sqrt(chi1x_hat)
            z_plus = -1.0 * (h1x - l) / np.sqrt(chi1x_hat)
            new_chi1x = (
                                norm.cdf(z_minus) + norm.sf(z_plus)
                        ) / q1x_hat
            return new_chi1x

        h1x = self.q1x_hat * self.r1x
        l = self.l
        chi1x_hat = self.chi1x_hat
        q1x_hat = self.q1x_hat
        new_chi1x = self.enlarge_prob * get_new_chi1x(chi1x_hat, h1x, l / self.enlarge_ratio, q1x_hat) + (
                1.0 - self.enlarge_prob) * get_new_chi1x(chi1x_hat, h1x, l, q1x_hat)

        return new_chi1x

    def _calculate_v1x(self):
        """ calculate v1x """

        def get_expectation_of_second_moment(chi1x_hat, h, l, q1x_hat):
            z_minus = -1.0 * (h + l) / np.sqrt(chi1x_hat)
            z_plus = -1.0 * (h - l) / np.sqrt(chi1x_hat)
            first_term = ((h + l) ** 2.0 + chi1x_hat) * norm.cdf(z_minus)
            second_term = -np.sqrt(chi1x_hat / 2.0 / np.pi) * (
                    2.0 * (h + l) + z_minus * np.sqrt(chi1x_hat)
            ) * np.exp(-z_minus ** 2.0 / 2.0)
            third_term = ((h - l) ** 2.0 + chi1x_hat) * norm.sf(z_plus)
            forth_term = np.sqrt(chi1x_hat / 2.0 / np.pi) * (
                    2.0 * (h - l) + z_plus * np.sqrt(chi1x_hat)
            ) * np.exp(-z_plus ** 2.0 / 2.0)
            return (
                           first_term + second_term + third_term + forth_term
                   ) / (q1x_hat ** 2.0)

        h = self.q1x_hat * self.r1x
        l = self.l
        chi1x_hat = self.chi1x_hat
        q1x_hat = self.q1x_hat

        expectation_of_second_moment = self.enlarge_prob * get_expectation_of_second_moment(chi1x_hat, h,
                                                                                            l / self.enlarge_ratio,
                                                                                            q1x_hat) + (
                                               1.0 - self.enlarge_prob) * get_expectation_of_second_moment(
            chi1x_hat, h, l, q1x_hat)

        return expectation_of_second_moment - (self._calculate_x1_hat() ** 2.0)

    def _calculate_u1_hat(self):
        return poisson_average(
            func=lambda c: c / (1.0 + c * self.q1u_hat),
            mu=self.mu
        ) * (self.y + self.q1u_hat * self.r1u)

    def _calculate_chi1u(self):
        return poisson_average(
            func=lambda c: c / (1.0 + c * self.q1u_hat),
            mu=self.mu
        )

    def _calculate_v1u(self):
        first_moment = poisson_average(
            func=lambda c: c / (1.0 + c * self.q1u_hat),
            mu=self.mu
        )
        second_moment = poisson_average(
            func=lambda c: (c / (1.0 + c * self.q1u_hat)) ** 2.0,
            mu=self.mu
        )
        first_term = self.chi1u_hat * second_moment
        second_term = (second_moment - first_moment ** 2.0) * ((self.y + self.q1u_hat * self.r1u) ** 2.0)
        return first_term + second_term

    # message from 1 to 2
    def calculate_message_from1to2(self):
        """ calculate message from variable node 1 to variable node 2 """
        # x
        self.q2x_hat = np.clip(a=self.eta1x_1 - self.q1x_hat, a_min=self.clip_min, a_max=self.clip_max)
        self.chi2x_hat = np.clip(a=self.eta1x_2 - self.chi1x_hat, a_min=self.clip_min, a_max=self.clip_max)
        self.r2x = (self.eta1x_1 * self.x1_hat - self.q1x_hat * self.r1x) / self.q2x_hat

        # u
        self.q2u_hat = np.clip(a=self.eta1u_1 - self.q1u_hat, a_min=self.clip_min, a_max=self.clip_max)
        self.chi2u_hat = np.clip(a=self.eta1u_2 - self.chi1u_hat, a_min=self.clip_min, a_max=self.clip_max)
        self.r2u = (self.eta1u_1 * self.u1_hat - self.q1u_hat * self.r1u) / self.q2u_hat

    # variable 2 estimation
    def variable2_estimation(self):
        """ estimate variables at variable node 2 """
        q_x_inv = 1.0 / self.q2x_hat
        b_x = np.linalg.inv(np.diag(self.q2u_hat) + (self.A * q_x_inv) @ self.AT)
        a_x = np.diag(q_x_inv) - q_x_inv.reshape(-1, 1) * self.AT @ b_x @ self.A * q_x_inv

        self.x2_hat = self._calculate_x2_hat(a_x)
        new_chi2x, new_v2x = self._calculate_chi2x(a=a_x), self._calculate_v2x(a=a_x)
        self.chi2x = np.clip(a=new_chi2x, a_min=self.clip_min, a_max=self.clip_max)
        self.v2x = np.clip(a=new_v2x, a_min=self.clip_min, a_max=self.clip_max)
        self.eta2x_1, self.eta2x_2 = 1.0 / self.chi2x, self.v2x / (self.chi2x ** 2.0)

        # u
        a_u = np.linalg.inv((self.A * (1.0 / self.q2x_hat)) @ self.AT + np.diag(self.q2u_hat))
        self.u2_hat = self._calculate_u2_hat(a_u)
        new_chi2u, new_v2u = self._calculate_chi2u(a=a_u), self._calculate_v2u(a=a_u)
        self.chi2u = np.clip(a=new_chi2u, a_min=self.clip_min, a_max=self.clip_max)
        self.v2u = np.clip(a=new_v2u, a_min=self.clip_min, a_max=self.clip_max)
        self.eta2u_1, self.eta2u_2 = 1.0 / self.chi2u, self.v2u / (self.chi2u ** 2.0)

    def _calculate_x2_hat(self, a):
        return a @ (self.q2x_hat * self.r2x + self.AT @ self.r2u)

    def _calculate_chi2x(self, a):
        return np.diag(a)

    def _calculate_v2x(self, a):
        def return_v1(a, chi2x_hat):
            return (a ** 2.0) @ chi2x_hat

        def return_v2(a, chi2u_hat, q2u_hat, AT):
            return ((a @ AT) ** 2.0) @ (chi2u_hat / q2u_hat ** 2.0)

        v1 = return_v1(a, self.chi2x_hat)
        v2 = return_v2(a, self.chi2u_hat, self.q2u_hat, self.AT)
        return v1 + v2

    def _calculate_u2_hat(self, a):
        b = - self.A @ self.r2x + self.q2u_hat * self.r2u
        return a @ b

    def _calculate_chi2u(self, a):
        return np.diag(a)

    def _calculate_v2u(self, a):
        v1 = (a ** 2.0) @ self.chi2u_hat
        v2 = (a @ self.A) ** 2.0 @ (self.chi2x_hat / self.q2x_hat ** 2.0)
        return v1 + v2

    # message from 2 to 1
    def calculate_message_from2to1(self):
        """ calculate message from variable node 2 to variable node 1 """
        # x
        self.q1x_hat = update_dumping(self.q1x_hat,
                                      np.clip(a=self.eta2x_1 - self.q2x_hat, a_min=self.clip_min, a_max=self.clip_max),
                                      self.eta)
        self.chi1x_hat = update_dumping(self.chi1x_hat,
                                        np.clip(a=self.eta2x_2 - self.chi2x_hat, a_min=self.clip_min,
                                                a_max=self.clip_max),
                                        self.eta)

        self.r1x = (self.eta2x_1 * self.x2_hat - self.q2x_hat * self.r2x) / self.q1x_hat

        # u
        self.q1u_hat = update_dumping(self.q1u_hat,
                                      np.clip(a=self.eta2u_1 - self.q2u_hat, a_min=self.clip_min, a_max=self.clip_max),
                                      self.eta)
        self.chi1u_hat = update_dumping(self.chi1u_hat,
                                        np.clip(a=self.eta2u_2 - self.chi2u_hat, a_min=self.clip_min,
                                                a_max=self.clip_max),
                                        self.eta)
        self.r1u = (self.eta2u_1 * self.u2_hat - self.q2u_hat * self.r2u) / self.q1u_hat

    def show_me(self):
        """ debug method """
        print("### x1 ###")
        print("self.chi1x_hat", self.chi1x_hat.mean())
        print("self.q1x_hat", self.q1x_hat.mean())
        print("self.chi1x", self.chi1x.mean())
        print("self.v1x", self.v1x.mean())
        print("### u1 ###")
        print("self.chi1u_hat", self.chi1u_hat.mean())
        print("self.q1u_hat", self.q1u_hat.mean())
        print("self.chi1u", self.chi1u.mean())
        print("self.v1u", self.v1u.mean())
        print("### x2 ###")
        print("self.chi2x_hat", self.chi2x_hat.mean())
        print("self.q2x_hat", self.q2x_hat.mean())
        print("self.chi2x", self.chi2x.mean())
        print("self.v2x", self.v2x.mean())
        print("### u2 ###")
        print("self.chi2u_hat", self.chi2u_hat.mean())
        print("self.q2u_hat", self.q2u_hat.mean())
        print("self.chi2u", self.chi2u.mean())
        print("self.v2u", self.v2u.mean())
        print()

    def return_variables(self):
        return [
            # ''' message from 2 to 1 '''
            # x
            self.r1x.mean(),
            self.chi1x_hat.mean(),
            self.q1x_hat.mean(),
            # u
            self.r1u.mean(),
            self.chi1u_hat.mean(),
            self.q1u_hat.mean(),
            # ''' variable 1 estimation '''
            # x
            self.x1_hat.mean(),
            self.chi1x.mean(),
            self.v1x.mean(),
            self.eta1x_1.mean(),
            self.eta1x_2.mean(),
            # u
            self.u1_hat.mean(),
            self.chi1u.mean(),
            self.v1u.mean(),
            self.eta1u_1.mean(),
            self.eta1u_2.mean(),
            # ''' message from 1 to 2 '''
            # x
            self.r2x.mean(),
            self.chi2x_hat.mean(),
            self.q2x_hat.mean(),
            # u
            self.r2u.mean().mean(),
            self.chi2u_hat.mean(),
            self.q2u_hat.mean(),
            # ''' variable 2 estimation '''
            # x
            self.x2_hat.mean().mean(),
            self.chi2x.mean(),
            self.v2x.mean(),
            self.eta2x_1.mean(),
            self.eta2x_2.mean(),
            # u
            self.u2_hat.mean(),
            self.chi2u.mean(),
            self.v2u.mean(),
            self.eta2u_1.mean(),
            self.eta2u_2.mean(),
        ]
