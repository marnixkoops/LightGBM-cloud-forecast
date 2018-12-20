import numpy as np


def mean_huber(y_true, y_pred, delta=1):
    huber = np.where(np.abs(y_true - y_pred) < delta, 0.5 * ((y_true - y_pred) ** 2),
                     delta * np.abs(y_true - y_pred) - 0.5 * (delta ** 2))
    return np.mean(huber)


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ape = np.abs((y_true - y_pred)) / (y_true + 1)
    return np.mean(ape)


def wmape(y_true, y_pred):
    wmape = np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1)
    return wmape

# Define modified weighted MAPE
# def mod_wmape(y_true, y_pred):
#     mod_wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.min([y_true, y_pred], axis=0))
#     return mod_wmape

# def mod_wmape2(y_true, y_pred):
#     loss = np.abs(y_true - y_pred) / \
#         np.max([np.min([y_true, y_pred], axis=0), np.ones(len(y_pred))], axis=0)
#     return np.mean(loss)

# Custom objectives
# def mod_wmape(y_pred, lgb_train):
#     y_true = lgb_train.get_label()
#     loss = np.abs(y_true - y_pred) / np.max([np.min([y_true, y_pred], axis=0), np.ones(len(y_pred))], axis=0)
#     return "mod_wmape", np.sum(loss), False

# def mod_wmape(y_pred, lgb_train):
#     y_true = lgb_train.get_label()
#     loss = np.sum(np.abs(y_true - y_pred)) / np.sum(np.min([y_true, y_pred], axis=0))
#     return 'mod_wmape', loss, False

# def log_cosh_obj(y_pred, lgb_train):
#     y_true = lgb_train.get_label()
#     x = y_pred - y_true
#     grad = np.tanh(x)
#     # hess = 1 / np.cosh(x)**2
#     hess = 1 - grad * grad
#     return grad, hess
#
# def log_cosh_loss(y_pred, lgb_train):
#     y_true = lgb_train.get_label()
#     x = y_pred - y_true
#     loss = np.log(np.cosh(x))
#     # format: name, metric, higher_is_better (boolean)
#     return 'log(hyperbolic cosine)', np.mean(loss), False
