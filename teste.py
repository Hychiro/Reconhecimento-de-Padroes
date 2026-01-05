from bciflow.datasets.cbcic import cbcic
from bciflow.datasets.bciciv2a import bciciv2a
from bciflow.datasets.bciciv2b import bciciv2b
from bciflow.modules.tf import filterbank
from bciflow.modules.sf.csp import csp
from bciflow.modules.fe.logpower import logpower
from bciflow.modules.fs import MIBIF
from bciflow.modules.clf import nbpw
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from codes.metrics import classification_metrics
for dataset_name in ['cbcic', 'bciciv2a']:
    if dataset_name == 'cbcic':
        dataset = cbcic
        path = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/CBCIC'
        maxSubjects = 11
    elif dataset_name == 'bciciv2a':
        dataset = bciciv2a
        path = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/2a'
        maxSubjects = 10
    elif dataset_name == 'bciciv2b':
        dataset = bciciv2b
        path = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/2b'
        maxSubjects = 10

    resultDict = {'FBCSP': {}, 'CSP': {}}

    for key in resultDict.keys():
        for subject in range(1, 2):  # sujeito
            classifiers = [
                lda(),
                nbpw(),
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42
                )
            ]

            for clf in classifiers:
                eegdata = dataset(subject=subject, path=path, labels=['left-hand', 'right-hand'])
                ws = 1.0
                startWind = int((eegdata['events']['cue'][0] + 0.5) * eegdata["sfreq"])
                stopWind = int(startWind + ws * eegdata["sfreq"])
                eegdata["X"] = eegdata["X"][:, :, :, startWind:stopWind]
                clf_name = clf.__class__.__name__
                if clf_name not in resultDict[key]:
                    resultDict[key][clf_name] = {}
                results = []
                if key == 'FBCSP':
                    eegdata = filterbank(eegdata, kind_bp='chebyshevII')

                    shape1,shape2,shape3,shape4 =  eegdata["X"].shape
                    # Criar DataFrame com X e y
                    X = eegdata["X"].reshape(len(eegdata["y"]), -1)  # flatten trials
            
                    y = eegdata["y"]
                    df = pd.DataFrame(X)
                    df["label"] = y

                    # Split 80/20
                    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                    n_train = int(0.8 * len(df))
                    df_train, df_test = df.iloc[:n_train], df.iloc[n_train:]

                    X_train = df_train.drop("label", axis=1).to_numpy()
                    y_train = df_train["label"].to_numpy()
                    X_train = X_train.reshape(len(y_train),shape2,shape3,shape4)

                    X_test = df_test.drop("label", axis=1).to_numpy()
                    y_test = df_test["label"].to_numpy()
                    X_test = X_test.reshape(len(y_test),shape2,shape3,shape4)
                    # Criar dois objetos eegdata (treino e teste)
                    eegdata_train = {"X": X_train, "y": y_train}
                    eegdata_test = {"X": X_test, "y": y_test}

                    # CSP
                    sf = csp()
                    sfResult_train = sf.fit_transform(eegdata_train)
                    sfResult_test = sf.transform(eegdata_test)

                    # LogPower
                    feResult_train = logpower(sfResult_train)
                    feResult_test = logpower(sfResult_test)

                    # Seleção de características
                    fs = MIBIF(8, clf=lda())
                    fsResult_train = fs.fit_transform(feResult_train)
                    fsResult_test = fs.transform(feResult_test)
                    # Treinar e avaliar classificador
                    clf.fit(fsResult_train["X"],fsResult_train["y"])
                    y_pred = clf.predict_proba(fsResult_test["X"])

                    for i in range(len(y_test)):
                        for classV, val in eegdata["y_dict"].items():
                            if val == fsResult_test["y"][i]:
                                true_labels = classV
                        results.append([1, (eegdata['events']['cue'][0] + 0.5), true_labels, *y_pred[i]])
                    results = np.array(results)
                    results = pd.DataFrame(results, columns=['fold', 'tmin', 'true_label', *eegdata['y_dict'].keys()])
                    filename = f"csvresults/simplifiedResults/{dataset_name}_{key}_{clf_name}_subject_{subject}_start_{eegdata['events']['cue'][0] + 0.5:.2f}_window_{ws:.2f}.csv".replace(".", "_")
                    results.to_csv(filename, index=False,columns=['fold', 'tmin', 'true_label', *eegdata["y_dict"].keys()])
                    
                elif key == 'CSP':
                    shape1,shape2,shape3,shape4 =  eegdata["X"].shape
                    # Criar DataFrame com X e y
                    X = eegdata["X"].reshape(len(eegdata["y"]), -1)  # flatten trials
            
                    y = eegdata["y"]
                    df = pd.DataFrame(X)
                    df["label"] = y

                    # Split 80/20
                    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                    n_train = int(0.8 * len(df))
                    df_train, df_test = df.iloc[:n_train], df.iloc[n_train:]

                    X_train = df_train.drop("label", axis=1).to_numpy()
                    y_train = df_train["label"].to_numpy()
                    X_train = X_train.reshape(len(y_train),shape2,shape3,shape4)

                    X_test = df_test.drop("label", axis=1).to_numpy()
                    y_test = df_test["label"].to_numpy()
                    X_test = X_test.reshape(len(y_test),shape2,shape3,shape4)
                

                    # Criar dois objetos eegdata (treino e teste)
                    eegdata_train = {"X": X_train, "y": y_train}
                    eegdata_test = {"X": X_test, "y": y_test}

                    # CSP
                    sf = csp()
                    sfResult_train = sf.fit_transform(eegdata_train)
                    sfResult_test = sf.transform(eegdata_test)

                    # LogPower
                    feResult_train = logpower(sfResult_train, flating=True)
                    feResult_test = logpower(sfResult_test, flating=True)

                    # Treinar e avaliar classificador
                    clf.fit(fsResult_train["X"],fsResult_train["y"])
                    y_pred = clf.predict_proba(fsResult_test["X"])

                    for i in range(len(y_test)):
                        for classV, val in eegdata["y_dict"].items():
                            if val == fsResult_test["y"][i]:
                                true_labels = classV
                        results.append([1, (eegdata['events']['cue'][0] + 0.5), true_labels, *y_pred[i]])
                    results = np.array(results)
                    results = pd.DataFrame(results, columns=['fold', 'tmin', 'true_label', *eegdata['y_dict'].keys()])
                    filename = f"csvresults/simplifiedResults/{dataset_name}_{key}_{clf_name}_subject_{subject}_start_{eegdata['events']['cue'][0] + 0.5:.2f}_window_{ws:.2f}.csv".replace(".", "_")
                    results.to_csv(filename, index=False,columns=['fold', 'tmin', 'true_label', *eegdata["y_dict"].keys()])
                    
