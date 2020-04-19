import unittest
import metrics

class TestMetrics(unittest.TestCase):

    def test_accuracy(self):
        true = [0,0,1,0,1,1]
        pred = [0,1,0,0,1,0]
        result = metrics.ClassificationMetrics()("accuracy", true, pred)
        self.assertEqual(result, 0.5)

    def test_acu(self):
        true = [0,0,1,0,1,1]
        pred = [0,1,0,0,1,0]
        pred_proba = [0.6, 0.6, 0.8, 0.6, 0.8, 0.4]
        result = metrics.ClassificationMetrics()('auc', true, pred, pred_proba)
        self.assertAlmostEqual(result,0.66666666)

    def test_f1(self):
        true = [0,0,1,0,1,1]
        pred = [0,1,0,0,1,0]
        result = metrics.ClassificationMetrics()("f1", true, pred)
        self.assertEqual(result, 0.4)

    def test_recall(self):
        true = [0,1,1,0,1,1]
        pred = [0,1,0,1,1,0]
        result = metrics.ClassificationMetrics()("recall", true, pred)
        self.assertEqual(result, 0.5)

    def test_precision(self):
        true = [0,1,1,0,1,1]
        pred = [0,1,0,1,1,0]
        result = metrics.ClassificationMetrics()("precision", true, pred)
        self.assertEqual(result, 0.6666666666666666)

    
    def test_logloss(self):
        true = [0,1,1,0,1,1]
        pred = [0,1,0,1,1,0]
        pred_proba = [0.6, 0.6, 0.8, 0.6, 0.8, 0.4]
        result = metrics.ClassificationMetrics()("logloss", true, pred, pred_proba)
        self.assertEqual(result, 0.6176641536694792)
    
        

        

    

