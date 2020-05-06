import numpy as np
from functools import reduce
from copy import deepcopy

from LCTM import ssvm
from LCTM import utils
import logging


logger = logging.getLogger(__name__)


def pretrain_weights(model, X, Y):
    # Take mean of all potentials
    n_samples = len(X)

    # Compute potential costs for each (correct) labeling
    costs = [ssvm.compute_costs(model, X[i], Y[i]) for i in range(n_samples)]
    costs = reduce(lambda x,y: x + y, costs)
    for key in costs:
        norms = np.linalg.norm(costs[key])
        # norms[norms==0] = 1
        costs[key] /= norms

    model.ws = costs


def subgradient_descent(
        model, X, Y, n_iter=100, C=1., pretrain=True, verbose=False,
        gradient_method="adagrad", learning_rate=0.1, decay_rate=.99, batch_size=5,
        update_period=25):

    if model.debug:
        np.random.seed(1234)

    n_samples = len(X)

    # Check that Xi is of size FxT
    # FIXME: This is probably a bug
    # if X[0].shape[0] > X[0].shape[1]:
    if X[0].shape[0] > X[0].shape[0]:
        X = [x.T for x in X]

    # if weights haven't been set yet then initialize
    if model.n_classes is None:
        model.n_features = X[0].shape[0]
        model.n_classes = np.max(list(map(np.max, Y))) + 1
        model.max_segs = utils.max_seg_count(Y)
        model.ws.init_weights(model)

        if pretrain:
            if model.is_latent:
                Z = [
                    utils.partition_latent_labels(Y[i], model.n_latent)
                    for i in range(n_samples)
                ]
                pretrain_weights(model, X, Z)
            else:
                pretrain_weights(model, X, Y)

    costs_truth = [ssvm.compute_costs(model, X[i], Y[i]) for i in range(n_samples)]
    # print("Unaries costs", [c['unary'].sum() for c in costs_truth])
    cache = deepcopy(costs_truth[0]) * 0.

    for t in range(n_iter):
        if gradient_method == "full gradient":
            batch_samples = np.arange(0, n_samples)
            batch_size = n_samples
        else:
            batch_samples = np.random.randint(0, n_samples, batch_size)

        # Compute gradient
        j = batch_samples[0]
        x = X[j]
        y = Y[j]
        costs = costs_truth[j]
        w_diff = ssvm.compute_ssvm_gradient(model, x, y, costs, C)
        for j in batch_samples[1:]:
            x = X[j]
            y = Y[j]
            costs = costs_truth[j]
            sample_grad = ssvm.compute_ssvm_gradient(model, x, y, costs, C)
            w_diff += sample_grad
        w_diff /= batch_size

        # === Weight Update ===
        # Vanilla SGD
        if gradient_method == "sgd" or gradient_method == "full gradient":
            eta = learning_rate * (1 - t / n_iter)
            w_diff = w_diff * eta
        # Adagrad
        elif gradient_method == "adagrad":
            cache += w_diff * w_diff
            w_diff = w_diff / (cache + 1e-8).sqrt() * learning_rate
        # RMSProp
        elif gradient_method == "rmsprop":
            # cache = decay_rate*cache + (1-decay_rate)*w_diff.^2
            if t == 0:
                cache += w_diff * w_diff
            else:
                cache *= decay_rate
                cache += w_diff * w_diff * (1 - decay_rate)
            w_diff = w_diff / np.sqrt(cache + 1e-8) * learning_rate

        model.ws -= w_diff

        # Print and compute objective
        if not (t + 1) % update_period:
            objective_new = np.mean(
                [model.objective(model.predict(X[i]), Y[i]) for i in batch_samples]
            )
            model.logger.objectives[t + 1] = objective_new
            if verbose:
                logger.info("Iter {}, obj={}".format(t + 1, objective_new))
