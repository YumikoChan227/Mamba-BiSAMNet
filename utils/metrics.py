import numpy as np


def _as_2d(array):
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    if array.ndim > 2:
        return array.reshape(array.shape[0], -1)
    return array


def SSD(y, y_pred):
    y = _as_2d(y)
    y_pred = _as_2d(y_pred)
    return np.sum(np.square(y - y_pred), axis=1)


def MAD(y, y_pred):
    y = _as_2d(y)
    y_pred = _as_2d(y_pred)
    return np.max(np.abs(y - y_pred), axis=1)


def PRD(y, y_pred):
    y = _as_2d(y)
    y_pred = _as_2d(y_pred)
    numerator = np.sum(np.square(y - y_pred), axis=1)
    denominator = np.sum(np.square(y), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * np.sqrt(numerator / denominator)


def COS_SIM(y, y_pred):
    y = _as_2d(y)
    y_pred = _as_2d(y_pred)
    numerator = np.sum(y * y_pred, axis=1)
    denominator = np.linalg.norm(y, axis=1) * np.linalg.norm(y_pred, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def SNR(y1, y2):
    y1 = _as_2d(y1)
    y2 = _as_2d(y2)
    signal_power = np.sum(np.square(y1), axis=1)
    noise_power = np.sum(np.square(y2 - y1), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(signal_power / noise_power)


def SNR_improvement(y_in, y_out, y_clean):
    return SNR(y_clean, y_out) - SNR(y_clean, y_in)
