'''
@Author: Yingshi Chen

@Date: 2020-03-10 14:09:28
@
# Description: 
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve,auc

def ROC_plot(X_,y_, pred_,title):
    nFeat = X_.shape[1]
    fpr_, tpr_, thresholds = roc_curve(y_, pred_)
    optimal_idx = np.argmax(tpr_ - fpr_)
#https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    optimal_threshold = thresholds[optimal_idx]
    auc_ = auc(fpr_, tpr_)
    title = "{} auc=".format(title)
    print("{} auc={} OT={:.4g}".format(title, auc_,optimal_threshold))
    plt.plot(fpr_, tpr_, label="{}:{:.4g}".format(title, auc_))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('SMPLEs={} Features={} OT={:.4g}'.format(X_.shape[0],nFeat,optimal_threshold))
    plt.legend(loc='best')
    plt.savefig(f"./_auc_[{nFeat}].jpg")
    plt.show()
    return auc_,optimal_threshold

def LossAtTest(model,data,lib,metric,config):
    info = {}
    if data.X_test is None:
        return info

    title = ""
    pred_val = model.predict(data.X_test)
    if metric=="auc":
        auc_,optimal_threshold = ROC_plot(data.X_test,data.y_test, pred_val,title)
        info["accuracy"] = auc_
        info["optimal_threshold"] = optimal_threshold
    elif metric == "l2":
        info["accuracy"] = ((data.y_test - pred_val) ** 2).mean()
    print(f'====== GBDT_learn={lib}\tstep: test={data.X_test.shape} ACCU@Test={info["accuracy"]:.5f}')
    return info