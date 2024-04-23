import sklearn.preprocessing
import sklearn.svm
import sklearn.linear_model
import sklearn.discriminant_analysis
import sklearn.neighbors
import sklearn.tree

def build_classifier(classifier_name):
    if classifier_name == 'svm':
        model = sklearn.svm.SVC(cache_size = 8192, gamma = 'auto')
    elif classifier_name == 'lr':
        model = sklearn.linear_model.LogisticRegression(
            solver='liblinear',
            penalty='l2',
        )
    elif classifier_name == 'lda':
        model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    elif classifier_name == 'knn':
        model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    elif classifier_name == 'dt':
        model = sklearn.tree.DecisionTreeClassifier()
    else:
        raise Exception('Invalid classifier')
        
    return model

def fit(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)
