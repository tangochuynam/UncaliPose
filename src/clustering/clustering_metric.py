"""
   This scipt defines functions for clustering evaluation.
   Author: Yan Xu, CMU
   Update: Jul 01, 2022
"""

try:
    from sklearn.metrics.cluster import pair_confusion_matrix
except ImportError:
    # Fallback for older sklearn versions
    def pair_confusion_matrix(labels_true, labels_pred):
        """Compute pair confusion matrix for older sklearn versions."""
        from sklearn.metrics.cluster import contingency_matrix
        contingency = contingency_matrix(labels_true, labels_pred)
        n_c = np.sum(contingency, axis=1, dtype=np.int64)
        n_k = np.sum(contingency, axis=0, dtype=np.int64)
        n = np.sum(n_c)
        
        # Compute pair confusion matrix
        sum_squares = np.sum(contingency**2)
        C = np.empty((2, 2), dtype=np.int64)
        C[1, 1] = int(sum_squares - n)
        C[0, 1] = int(np.sum(n_k**2) - sum_squares)
        C[1, 0] = int(np.sum(n_c**2) - sum_squares)
        C[0, 0] = int(n**2 - C[0, 1] - C[1, 0] - sum_squares)
        return C

import numpy as np


def getPurity(label_true, label_pred):
    '''
    Compute the purity score.
    '''
    clusters = np.unique(label_pred)
    label_true = np.reshape(label_true, (-1, 1))
    label_pred = np.reshape(label_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(label_pred == c)[0]
        labels_tmp = label_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    purity = np.sum(count) / label_true.shape[0]
    return purity


def getRandIndexAndFScore(label_true, label_pred, beta=1.):
    '''
    Compute the random index and the adjusted random index.
    '''
    (tn, fp), (fn, tp) = pair_confusion_matrix(label_true, label_pred)
    ri = (tp + tn) / (tp + tn + fp + fn)
    if (tp+fn)*(fn+tn)+(tp+fp)*(fp+tn) == 0:
        ari = np.nan
    else:
        ari = 2.*(tp*tn-fn*fp)/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_score = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    return ri, ari, f_score


def getClustMetrics(label_true, label_pred):
    '''
    Get all clustering metrics and return a metric dictionary.
    '''
    purity = getPurity(label_true, label_pred)
    ri, ari, f_score = getRandIndexAndFScore(label_true, label_pred, beta=1.)
    return { 'Purity': purity, 'RI': ri, 'ARI': ari, 'F_Score': f_score}
