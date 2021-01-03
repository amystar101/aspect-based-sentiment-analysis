from sklearn import svm
from sklearn import metrics
import numpy as np


# using the svm function from sklearn

def feature_conversion(features,mp):
    x = np.zeros((len(features),len(mp.keys())))
    for i in range(0,len(features)):
        for j in features[i]:
            x[i][mp[j]] = 1
    return x;


def sk_svm(raw_feature,y,mp):
    print("Applying support vector machine from sklearn now...")
    x = feature_conversion(raw_feature,mp)
    clf = svm.SVC(kernel="rbf")

    clf.fit(x,y)
    y_pred = clf.predict(x)

    print("sklearn svm results :-")

    print("Accuracy:",metrics.accuracy_score(y, y_pred))
    print("Precision:",metrics.precision_score(y, y_pred))
    print("Recall:",metrics.recall_score(y, y_pred))


    return clf


def performance(model,raw_feature,y,mp):
    x = feature_conversion(raw_feature,mp)

    y_pred = model.predict(x)

    print("Accuracy:",metrics.accuracy_score(y, y_pred))
    print("Precision:",metrics.precision_score(y, y_pred))
    print("Recall:",metrics.recall_score(y, y_pred))
