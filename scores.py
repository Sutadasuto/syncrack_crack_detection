import cv2
import math
import numpy as np
import os
import warnings

from scipy import stats
from skimage.feature import greycomatrix as glcm


def calculate_dsc(y, y_pred):
    intersection = y_pred * y
    return (2 * np.sum(intersection) + 1) / (np.sum(y_pred) + np.sum(y) + 1)


def calculate_confusion_matrix(y, y_pred):
    confusion_matrix = np.zeros((2, 2))
    matrix = np.array([['true_positives', 'false_positives'], ['false_negatives', 'true_negatives']])
    h, w = y.shape
    tp = y_pred * y
    fp = y_pred - tp
    fn = np.maximum(0, (1 - y_pred) - (1 - y))

    confusion_matrix[0, 0] = np.sum(tp)
    confusion_matrix[0, 1] = np.sum(fp)
    confusion_matrix[1, 0] = np.sum(fn)
    confusion_matrix[1, 1] = h * w - np.sum(tp) - np.sum(fp) - np.sum(fn)

    return confusion_matrix, matrix


def calculate_PRF(confusion_matrix):
    if not confusion_matrix[0, 0] + confusion_matrix[0, 1] == 0:
        precision = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
    else:
        if confusion_matrix[1, 0] == 0:
            warnings.warn(
                'The sum of true and false positives is 0 (no pixel predicted as crack). However, no false negative is present: setting precision to 1.')
            precision = 1
        else:
            warnings.warn(
                'The sum of true and false positives is 0 (no pixel predicted as crack). False negatives are present: setting precision to 0.')
            precision = 0
    if not confusion_matrix[0, 0] + confusion_matrix[1, 0] == 0:
        recall = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
    else:
        warnings.warn(
            'The sum of true positives and false negatives is 0 (no crack pixel exists in the groundtruth). Setting recall to 1.')
        recall = 1
    if precision == 0 and recall == 0:
        warnings.warn('Precision and recall equal to 0. Setting f-score to 0 (totally wrong prediction).')
        f = 0
    else:
        f = 2 * precision * recall / (precision + recall)
    return precision, recall, f


def calculate_Hr(x, y_pred):
    Rj = y_pred
    Sj = np.sum(Rj)
    Vj, Lj = np.unique(x[np.where(Rj == 1.0)], return_counts=True)

    Hr = 0
    for m in range(len(Vj)):
        Hr += (Lj[m] / Sj) * math.log(Lj[m] / Sj)
    Hr *= -1
    return Hr


def calculate_approximate_Hr2(x, y_pred):
    # Calculate an approximation of the second order entropy inside a mask.
    # To calculate the intra-mask co-occurrence matrix, the input image is converted to uint16. The image pixels outside
    # the mask are assigned 256 as intensity and the co-occurrence matrix is calculated with intensities in the range
    # [0, 256]. Then, the last row and column from the matrix are suppressed.
    # The co-occurrence probability matrix is calculated with a (radius 1) 8-pixels neighborhood

    intensities = 2**8
    if x.dtype != np.uint8:
        x = (255*x).astype(np.uint8)
    x_16b = x.astype(np.uint16)
    x_16b[np.where(y_pred == 0)] = intensities
    cm = glcm(x_16b, [1], [i*np.pi/4 for i in range(8)], intensities+1)
    cm = cm[:-1, :-1]  # Crop the row and column corresponding to intensity 256
    cm = np.sum(cm[...,0,:], axis=-1) # Join the 8 directions into a single 2D matrix
    p = cm.astype(np.float64)/np.sum(cm)

    Hr2 = 0
    for i in range(intensities):
        for j in range(intensities):
            if p[i, j] > 0:
                Hr2 += p[i, j] * math.log(p[i, j])
    Hr2 *= -1
    return Hr2


def calculate_kolmogorov_smirnov_statistic(x, y_pred, alpha):
    bkgd = x[y_pred == 0]
    crack = x[y_pred != 0]
    if len(crack) == 0 or len(bkgd) == 0:
        warnings.warn('One of the classes is not populated. Thus, the whole image is classified as crack or background.'
                      ' Since the network did not make a difference between the two distributions (classes), they are'
                      ' assumed to be the same one. Therefore the distance between them is assumed to be 0.')
        statistic, p_value = (0.0, 0.0)
    else:
        statistic, p_value = stats.ks_2samp(bkgd, crack)
    if p_value > alpha:
        warnings.warn(
            'The p-value {:.4f} is greater than the alpha {:.4f}. We cannot reject the null hypothesis in favor of '
            'the alternative, thus we assume the Kolmogorov-Smirnov statistic to be zero (the distributions are '
            'identical).')
        statistic = 0.0
    return statistic
