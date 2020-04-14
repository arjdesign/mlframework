Simple and Binary Classification Metircs:

-Binary Classifictaion
    metrics: 
    -Accuracy
    -Precision
    -Recall
    -F1-Score(f1)
    -AUC (Area under the Receiver Operating Characterstics (ROC) curve)
    -log loss

    True Postitves (TP): Prediction is positive and the actual is positive too
    True Negative (TN): Prediction is negativa and the actual is negative too.
    False Positive (FP): Prediction is positive but the actual value is negative
    False Negative (FN): Prediction is negative but the actual value is positive

    Accuracy: = (true positive + true Negative)/(Total observations)
                (TP +TN)/(TP+TN+FP+FN)

    precision recall tradeoff 
            TN FP
            FN TP
    precision = TP/(TP+FP)
    High precision means you are prediction less false positive

    RECALL = TP/(FN+FP)
    High recall means you are predicting less false negativates
    Your recall should be higher than 0.5 and if it is towards 1 its very good. 
    Precision and reacall both are between 0-1. Higher the value better it is.

F1 Is the weighted aveage of precision and recall 

F1 = 2*(precision *recall)
    ----------------------
     (precision+Recall)  

          2TP
     ---------------
     (2TP + FP + FN)

F1 is the harmonic average of precision and recall. It is the good measure of model's accuracy. Its value ranges from 0-1


AUC: 1.0 good model. Auc 0.5 is random model
Given the postive random sample and the negative random sample in the  dataset, when is the probability that the postive sample will rank higher than negative sample. That 
Probability value is represented by AUC. For AUC you do not have to choose the probability all the time you have to choose th cut off number and make a plot.

TPR = 
            |       /
            |      /
 TPR        |     /
 (Recall    |    /
Sensitivity)|   /
            |  /
            | /
            |/_________________
               False Postive rate 



TPR = TP/(TP + FN) In another word TP/ (Actual Positive numbers)
FPR = FP/(FP+FN) This is ==> 1-specificity

9:47 MIn
https://www.youtube.com/watch?v=uuQ4XLeyWG0&list=PL98nY_tJQXZnKfgWIADbBG182nFUNIsxw&index=5


LogLoss:

-1 * (ylog(p)+ (1-y)log(1-p))

0      1
-----|-----
0.1  | 0.9
0.1  | 0.6

If the actual value is 1 and prediction is 0.6, logloss assigns more penatly to it compared to 0.9
If you are confident and wrong, you are penalized more.

Lower Logloss value is better.

For every predition loglossis calculated. Final logloss is given by averaging all the loglosses.

