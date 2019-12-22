import numpy as np
import time

class MF():
    def __init__(self, R, K, alpha, beta, target_accuracy, max_iterations, tol, logger):
        """
        Perform matrix factorization to predict empty
        values in a matrix.

        Arguments
        - R (ndarray)     : user-item rating full matrix
        - K (int)         : number of latent dimensions
        - alpha (float)   : learning rate
        - beta (float)    : regularization parameter
        - target_accuracy : training stops when target accuracy is reached
        - max_iterations  : training stops when max iteration is reached
        - tol         : training stops when error difference between two consecutive iteration is less than tol
        """

        self.R = R
        self.user_indexes, self.item_indexes = self.R.nonzero()
        self.ground_truth = self.R[self.user_indexes, self.item_indexes]
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.target_accuracy = target_accuracy
        self.max_iterations = max_iterations
        self.tol = tol
        self.logger = logger

    def train(self):
        start_time = time.time()

        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.ground_truth)

        # Create a list of training samples
        self.samples = list(zip(self.user_indexes, self.item_indexes, self.ground_truth))

        self.logger.log("Number of training samples: {}\n".format(len(self.samples)))

        rmses = []
        # Perform stochastic gradient descent for number of iterations
        for i in range(self.max_iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse = self.rmse()
            rmses.append(rmse)

            if (len(rmses) > 10) and ((rmses[i-10] - rmses[i]) < self.tol):
                self.logger.log("Iteration: %d ; error = %.4f" % (i+1, rmse))
                self.logger.log("Target error difference (tol) {} reached.".format(str(self.tol)))
                break

            if rmse < 1 - self.target_accuracy:
                self.logger.log("Iteration: %d ; error = %.4f" % (i+1, rmse))
                self.logger.log("Target accuracy of {} reached.".format(str(self.target_accuracy)))
                break

            if (i+1) % 10 == 0:
                self.logger.log("Iteration: %d ; error = %.4f" % (i+1, rmse))

        end_time = time.time()
        training_time = str(round(end_time - start_time, 4))

        self.logger.log("Training time: {}s".format(training_time))

    def rmse(self):
        """
        A function to compute the total root mean square error
        """
        predictions = self.full_matrix()[self.user_indexes, self.item_indexes]
        return np.sqrt(np.square(self.ground_truth - predictions).mean())

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_engagement_rate(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Simultaneously update user and item latent feature matrices
            temp_P = self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            temp_Q = self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

            self.P[i, :] += temp_P
            self.Q[j, :] += temp_Q

    def get_engagement_rate(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Compute the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)

    def parameters(self):
        output = {}

        output['b'] = self.b
        output['b_i'] = np.array(self.b_i)
        output['b_u'] = np.array(self.b_u)
        output['P'] = np.array(self.P)
        output['Q'] = np.array(self.Q)

        return output

    def log_hyperparameters(self):
        self.logger.log("")
        self.logger.log("K: {}".format(self.K))
        self.logger.log("alpha (learning rate): {}".format(self.alpha))
        self.logger.log("beta (regularization): {}".format(self.beta))
        self.logger.log("target_accuracy: {}".format(self.target_accuracy))
        self.logger.log("max_iterations: {}".format(self.max_iterations))
        self.logger.log("tol: {}".format(self.tol))

