from sklearn import metrics as skmetrics

class ClassificationMetrics:
    def __init__(self):
        self.metrics = {
            "accuracy":self._accuracy,
            "auc": self._auc,
            "f1": self._f1,
            "recall": self._recall,
            "precision": self._precision
        }

    def __call__(self, metric, y_true, y_pred, y_proba=None):
        
        if metric not in self.metrics:
            raise Exception("metrics not implemented")
        
        return self.metrics[metric](y_true=y_true, y_pred=y_pred )
        


    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true= y_true, y_pred = y_pred)

    @staticmethod
    def _auc(y_true, y_pred):
        return skmetrics.roc_auc_score(y_true = y_true, y_score = y_pred)
    
    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _recall (y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _precision (y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _logloss (y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)
    

    
    