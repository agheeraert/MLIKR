from sklearn import tree
import pickle as pkl
import numpy as np

def load(path):
    return pkl.load(open(path, 'rb'))

X_test = load('X_test.p')
X_train = load('X_train.p')
Y_test = load('Y_test.p')
Y_train = load('Y_train.p')

clf = tree.DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, Y_train)

acc_decision_tree = clf.score(X_test, Y_test)*100
print(acc_decision_tree, "%")
print(np.argsort(clf.feature_importances_)[-1])
