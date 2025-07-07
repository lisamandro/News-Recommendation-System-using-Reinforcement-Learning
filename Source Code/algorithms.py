import numpy as np
import dataset

#LinUCB with Contextual features
class LinUCB:

    def __init__(self, alpha, context="user"):
        """
        Parameters
        ----------
        alpha : float
            LinUCB parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        """
        self.n_features = len(dataset.features[0])
        if context == "user":
            self.context = 1
        elif context == "both":
            self.context = 2
            self.n_features *= 2

        self.A = np.array([np.identity(self.n_features)] * dataset.n_arms)
        self.A_inv = np.array([np.identity(self.n_features)] * dataset.n_arms)
        self.b = np.zeros((dataset.n_arms, self.n_features, 1))
        self.alpha = round(alpha, 1)
        self.algorithm = "LinUCB (α=" + str(self.alpha) + ", context:" + context + ")"

    def select_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        A_inv = self.A_inv[pool_idx]
        b = self.b[pool_idx]

        n_pool = len(pool_idx)

        user = np.array([user] * n_pool)
        if self.context == 1:
            x = user
        else:
            x = np.hstack((user, dataset.features[pool_idx]))

        x = x.reshape(n_pool, self.n_features, 1)

        theta = A_inv @ b

        p = np.transpose(theta, (0, 2, 1)) @ x + self.alpha * np.sqrt(
            np.transpose(x, (0, 2, 1)) @ A_inv @ x
        )
        return np.argmax(p)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        arm = pool_idx[displayed]  # displayed article's index
        if self.context == 1:
            x = np.array(user)
        else:
            x = np.hstack((user, dataset.features[arm]))

        x = x.reshape((self.n_features, 1))

        self.A[arm] += x @ x.T
        self.b[arm] += reward * x
        self.A_inv[arm] = np.linalg.inv(self.A[arm])

#Simple Upper Confidence Bound
class UCB:
    """
    UCB algorithm implementation
    """

    def __init__(self, alpha):
        """
        Parameters
        ----------
        alpha : number
            ucb parameter
        """

        self.alpha = round(alpha, 1)
        self.algorithm = "UCB1 (α=" + str(self.alpha) + ")"

        self.q = np.zeros(dataset.n_arms)  # average reward for each arm
        self.n = np.ones(dataset.n_arms)  # number of times each arm was chosen

    def select_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        ucbs = self.q[pool_idx] + np.sqrt(self.alpha * np.log(t + 1) / self.n[pool_idx])
        return np.argmax(ucbs)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        arm = pool_idx[displayed]

        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]

class ThompsonSampling:
    """
    Thompson sampling algorithm implementation
    """

    def __init__(self):
        self.algorithm = "TS"
        self.alpha = np.ones(dataset.n_arms)
        self.beta = np.ones(dataset.n_arms)

    def select_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        theta = np.random.beta(self.alpha[pool_idx], self.beta[pool_idx])
        return np.argmax(theta)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        arm = pool_idx[displayed]

        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

class EpsilonGreedy:

    def __init__(self, epsilon):
        """
        Parameters
        ----------
        epsilon : number
            E-Greedy parameter
        """

        self.e = round(epsilon, 1)  # epsilon parameter for E-Greedy
        self.algorithm = "Egreedy (ε=" + str(self.e) + ")"
        self.q = np.zeros(dataset.n_arms)  # average reward for each arm
        self.n = np.zeros(dataset.n_arms)  # number of times each arm was chosen

    def select_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        p = np.random.rand()
        if p > self.e:
            return np.argmax(self.q[pool_idx])
        else:
            return np.random.randint(low=0, high=len(pool_idx))

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        arm = pool_idx[displayed]

        self.n[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.n[arm]