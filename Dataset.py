from utils.featurizer import featurize_all, DihedralFeaturizer, AlphaAngleFeaturizer
from os import listdir
from os.path import join as jn
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from Bio.PDB.Polypeptide import aa1, aa3

three2one = dict(zip(aa3, aa1))
BASE_FOLDER = '/home/agheerae/Python/MLIKR'

def dump(obj, path):
    return pickle.dump(obj, open(path, 'wb'))


filenames_train = [jn(BASE_FOLDER, 'data/dcd/train', elt) for elt in listdir('data/dcd/train')]
filenames_test = [jn(BASE_FOLDER, 'data/dcd/test', elt) for elt in listdir('data/dcd/test')]

#Featurizing the dihedral angles
psi_features_train = featurize_all(filenames_train, DihedralFeaturizer, jn(BASE_FOLDER, 'data/prmtop/prot.prmtop'), chunk=1000) 
psi_features_test = featurize_all(filenames_test, DihedralFeaturizer, jn(BASE_FOLDER, 'data/prmtop/prot.prmtop'), chunk=1000) 


# Retrieving values (bound=1, unbound=0)
Y_train = []
for elt in psi_features_train[2]:
    if elt[-14] == 'r':
        Y_train.append(1)    
    elif elt[-14] == 'o':
        Y_train.append(0)
Y_test = []
for elt in psi_features_test[2]:
    if elt[-14] == 'r':
        Y_test.append(1)    
    elif elt[-14] == 'o':
        Y_test.append(0)

#Featurizing the alpha
alpha_features_train = featurize_all(filenames_train, AlphaAngleFeaturizer, jn(BASE_FOLDER, 'data/prmtop/prot.prmtop'), chunk=1000)
alpha_features_test = featurize_all(filenames_test, AlphaAngleFeaturizer, jn(BASE_FOLDER, 'data/prmtop/prot.prmtop'), chunk=1000)
X_train = np.concatenate((psi_features_train[0], alpha_features_train[0]), axis=1)
n_alpha_features, n_psi_phi_features = int(len(alpha_features_test[3])/2), int(len(psi_features_test[3])/2)
X_train_featdic = np.concatenate((psi_features_test[3][:n_psi_phi_features], alpha_features_test[3][:n_alpha_features]))
X_test = np.concatenate((psi_features_test[0], alpha_features_test[0]), axis=1)

rf = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

rf = rf.fit(X_train, Y_train)

acc_decision_tree = rf.score(X_test, Y_test)*100
print(acc_decision_tree, "%")
X_train_feattype = []
X_train_featres = []

for elt in X_train_featdic:
    if elt['featurizer'] == 'AlphaAngle':
        X_train_feattype.append('alpha')
    else:
        X_train_feattype.append(elt['featuregroup'])
    if elt['resseqs'][0] < 253:
        chain = 'f'
    else:
        chain = 'h'
    L_residues = []
    for i in range(len(elt['resseqs'])):
        L_residues.append(chain+three2one[elt['resnames'][i]]+str((elt['resids'][i]%253)+1))
    X_train_featres.append(L_residues)

    


importances = pd.DataFrame({'type':X_train_feattype, 'residues':X_train_featres, 'importance':np.round(rf.feature_importances_,3)*100})
importances = importances.sort_values('importance',ascending=False).set_index('type')
importances.to_excel('random_forest_results.xlsx')