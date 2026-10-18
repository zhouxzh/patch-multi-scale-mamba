import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def _safe_scale(scale, eps=1e-12):
    scale = np.asarray(scale, dtype=np.float64)
    return np.maximum(scale, eps)


def RMSSE(pred, true, insample, seasonality=1):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    insample = np.asarray(insample, dtype=np.float64)

    if seasonality < 1:
        raise ValueError("seasonality must be >= 1")
    if insample.shape[0] <= seasonality:
        raise ValueError("insample series is too short for RMSSE")

    naive_diff = insample[seasonality:] - insample[:-seasonality]
    scale = _safe_scale(np.mean(np.square(naive_diff), axis=0))
    scaled_sq_error = np.square(pred - true) / scale
    return np.sqrt(np.mean(scaled_sq_error))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true, insample=None, seasonality=1):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rmsse = RMSSE(pred, true, insample, seasonality) if insample is not None else None

    return mae, mse, rmse, mape, mspe, rmsse
