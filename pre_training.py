from sklearn import tree
import pickle as pkl

def load(path):
    return pkl.load(open(path, 'rb'))

X_test = load('X_test.p')
X_train = load('X_train.p')
Y_test = load('Y_test.p')
Y_train = load('Y_train.p')

clf = tree.DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

acc_decision_tree = round(clf.score(X_train, Y_train) * 100, 2)
print(round(acc_decision_tree,2,), "%")
