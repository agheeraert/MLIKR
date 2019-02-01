from utils.featurizer import featurize_all, DihedralFeaturizer, BinaryContactFeaturizer, ContactFeaturizer
from os import listdir
from os.path import join as jn
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

BASE_FOLDER = '/home/agheerae/Python/MLIKR'

def dump(obj, path):
    return pickle.dump(obj, open(path, 'wb'))

filenames = [jn(BASE_FOLDER, 'data/dcd', elt) for elt in listdir('data/dcd')]
#Featurizing the angles
psi_features = featurize_all(filenames, DihedralFeaturizer, jn(BASE_FOLDER, 'data/prmtop/prot.prmtop'), chunk=1000) 
# Retrieving values (bound=1, unbound=0)
Y_tot = []
for elt in psi_features[2]:
    if elt[-14] == 'r':
        Y_tot.append(1)    
    elif elt[-14] == 'o':
        Y_tot.append(0)
print('Featurizing distances')
#Featurizing the distances
d_features = featurize_all(filenames, BinaryContactFeaturizer, jn(BASE_FOLDER, 'data/prmtop/prot.prmtop'), chunk=125)
print(d_features)
X_tot = np.concatenate((psi_features[0], d_features[0]), axis=1)
print(X_tot)

#Splitting in random train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_tot, Y_tot, test_size=0.33, random_state=42)

#Dumping
dump(X_train, 'X_train.p')
dump(X_test, 'X_test.p')
dump(Y_train, 'Y_train.p')
dump(Y_test, 'Y_test.p')


