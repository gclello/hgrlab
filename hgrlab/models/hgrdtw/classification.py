import sklearn.preprocessing
import sklearn.svm
import sklearn.linear_model
import sklearn.discriminant_analysis
import sklearn.neighbors
import sklearn.tree

def build_classifier(classifier_name, classifier_options={}):
    def merge_options(options1, options2):
        options1 = {} if options1 is None else options1
        options2 = {} if options2 is None else options2
        for option in options2:
            options1[option] = options2[option]
        return options1

    if classifier_name == 'svm':
        model = sklearn.svm.SVC(**merge_options({
            'cache_size': 8192,
            'gamma': 'auto',
            },
            classifier_options,
        ))
    elif classifier_name == 'lr':
        model = sklearn.linear_model.LogisticRegression(**merge_options({
                'solver': 'liblinear',
                'penalty': 'l2',
            },
            classifier_options,
        ))
    elif classifier_name == 'lda':
        model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
            **merge_options(
                {'solver': 'svd'},
                classifier_options,
            )
        )
    elif classifier_name == 'knn':
        model = sklearn.neighbors.KNeighborsClassifier(**merge_options({
                'n_neighbors': 5,
            },
            classifier_options
        ))
    elif classifier_name == 'dt':
        model = sklearn.tree.DecisionTreeClassifier(
            **merge_options(
                {},
                classifier_options,
            )
        )
    else:
        raise Exception('Invalid classifier')
    
    return model

def fit(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)
