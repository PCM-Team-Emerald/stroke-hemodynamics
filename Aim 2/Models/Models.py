from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


class LogisticRegressionModel():
    def __init__(self):
        self.params = {
            'penalty' : 'l2',
            'solver' : 'lbfgs',
            'max_iter' : 100,
            'verbose' : 0
        }
        self.model = LogisticRegression(**self.params)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        raw_predictions = self.model.predict_proba(X_test)
        score = self.model.score(X_test, y_test)
        return raw_predictions, predictions, score

    
class RandomForestModel():
    def __init__(self):
        self.params = {
            'n_estimators' : 100,
            'criterion' : 'gini',
            'max_depth' : None,
            'verbose' : 0
        }
        self.model = RandomForestClassifier(**self.params)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        raw_predictions = self.model.predict_proba(X_test)
        score = self.model.score(X_test, y_test)
        return raw_predictions, predictions, score

    
def metrics(y_test, raw_predictions, predictions):
    fpr, tpr, _ = roc_curve(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    
    precisions, recalls, _ = precision_recall_curve(y_test, predictions)
    
    plt.figure(figsize = (12,6))
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(['AUC = {}'.format(auc)], loc='lower right')
    
    plt.subplot(1,2,2)
    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
