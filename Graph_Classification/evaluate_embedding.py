from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
import numpy as np

def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(random_state=42), params, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)

def evaluate_embedding(embeddings, labels, search=True):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    svc_accuracies = [svc_classify(x,y, search) for _ in range(1)]

    return np.mean(svc_accuracies) 
