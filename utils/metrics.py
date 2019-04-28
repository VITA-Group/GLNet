# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

class ConfusionMatrix(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        # axis = 0: target
        # axis = 1: prediction
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        # self.iou = []
        # self.iou_threshold = []

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            tmp = self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
            
            # iu = np.diag(tmp) / (tmp.sum(axis=1) + tmp.sum(axis=0) - np.diag(tmp))
            # self.iou.append(iu[1])
            # if iu[1] >= 0.65: self.iou_threshold.append(iu[1])
            # else: self.iou_threshold.append(0)
            
            self.confusion_matrix += tmp

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        # accuracy is recall/sensitivity for each class, predicted TP / all real positives
        # axis in sum: perform summation along
        acc = np.nan_to_num(np.diag(hist) / hist.sum(axis=1))
        acc_mean = np.mean(np.nan_to_num(acc))
        
        intersect = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        iou = intersect / union
        mean_iou = np.mean(np.nan_to_num(iou))
        
        freq = hist.sum(axis=1) / hist.sum() # freq of each target
        # fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        freq_iou = (freq * iou).sum()

        return {'accuracy': acc,
                'accuracy_mean': acc_mean,
                'freqw_iou': freq_iou,
                'iou': iou, 
                'iou_mean': mean_iou, 
                # 'IoU_threshold': np.mean(np.nan_to_num(self.iou_threshold)),
                }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        # self.iou = []
        # self.iou_threshold = []