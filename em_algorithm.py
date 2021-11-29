from numpy import *
import pandas as pd
import numpy as np

# load data as numpy array
def load_data(data_set):
    X = mat(data_set.T[:].T)
    sample_count, feature_count = shape(X)
    avg_set = [average(col) for col in data_set.T[:]]

    return X, sample_count, feature_count, avg_set

# init parameters
def init_par(k, X, init_phi, init_mu, init_cov, feature_count):
    # init phi distribution
    for i in range(k):
        init_phi.append(1 / k)

    # randomly select k sample as initial mu of k class
    init_mu = [X[i, :] for i in random.randint(0,150,size=k)]

    # init identity matrix as initial covariance matrix
    init_cov = [mat(np.identity(feature_count)) for _ in range(k)]
    
    return init_phi, init_mu, init_cov

# calculate multivariate Normal distribution probabilities of each sample(gamma)
def G_prob(x, mu, cov):
    n = len(x[0])
    e_power = float(-0.5 * (x - mu) * (cov.I) * ((x - mu).T))
    Deno = power(2 * pi, n / 2) * power(linalg.det(cov), 0.5)
    gamma = power(e, e_power) / Deno
    return gamma

# implement EM Algorithm
def EM(X, init_phi, init_cov, init_mu, k, sample_count, feature_count):

    # init parameters
    phi = init_phi
    cov = init_cov
    mu = init_mu

    # init probabilities set
    gamma = mat(zeros((sample_count, k)))

    # Start Iteration
    dif = 1
    threshold = 1e-3
    while dif > threshold:
        mu_pre = [item for item in mu]
        # step E
        for j in range(sample_count):
            px = 0
            for i in range(k):
                gamma[j, i] = phi[i] * G_prob(X[j, :], mu[i], cov[i])
                px += gamma[j, i]
            for i in range(k):
                gamma[j, i] /= px
        sum_gamma = sum(gamma, axis=0)

        # step M
        for i in range(k):
            mu[i] = mat(zeros((1, feature_count)))
            cov[i] = mat(zeros((feature_count, feature_count)))
            for j in range(sample_count):
                mu[i] += gamma[j, i] * X[j, :]
            mu[i] /= sum_gamma[0, i]
            for j in range(sample_count):
                cov[i] += gamma[j, i] * (X[j, :] - mu[i]).T * (X[j, :] - mu[i])
            cov[i] /= sum_gamma[0, i]
            phi[i] = sum_gamma[0, i] / sample_count

        # check whether mu are convergence
        dif = 0
        for i in range(k):
            distance = (mu[i]-mu_pre[i])*(mu[i]-mu_pre[i]).T
            dif += distance[0,0]
    return gamma

# cluster samples to k groups
def cluster(X, init_phi, init_cov, init_mu, k, sample_count, feature_count, result, centroids):
    # init centroids set for different classes
    gamma = EM(X, init_phi, init_cov, init_mu, k, sample_count, feature_count)
    classification = mat(zeros((sample_count, 2)))


    for i in range(sample_count):
        # Align to groups (return the index of biggest probability, and such prob)
        classification[i, :] = argmax(gamma[i, :]), amax(gamma[i, :])
        temp = [item for item in squeeze(np.asarray(X[i, :]))] + [argmax(gamma[i, :]), amax(gamma[i, :])]
        result.loc[i] = temp

        # update centroids
    for j in range(k):
        pointsInCluster = X[nonzero(classification[:, 0].A == j)[0]]
        centroids.append(mean(pointsInCluster, axis=0))

    # set 'class' column data type to int
    result['class'] = pd.to_numeric(result['class'], downcast='signed', errors='coerce')
    
    return result, centroids