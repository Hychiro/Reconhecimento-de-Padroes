from bciflow.datasets.cbcic import cbcic
from bciflow.datasets.bciciv2a import bciciv2a
from bciflow.datasets.bciciv2b import bciciv2b
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.bandpass import bandpass_conv,chebyshevII
from bciflow.modules.tf import  filterbank 
from bciflow.modules.fs import MIBIF
from bciflow.modules.sf.csp import csp
from bciflow.modules.fe.logpower import logpower
from bciflow.modules.analysis import metric_functions
from bciflow.modules.clf import nbpw
from eegnet import MBEEGNet
import numpy as np
import time
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import matplotlib.pyplot as plt
from codes.metrics import classification_metrics
from xgboost import XGBClassifier



for dataset_name in ['cbcic', 'bciciv2a', 'bciciv2b']:
    if dataset_name == 'cbcic':
        dataset = cbcic
        path = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/CBCIC'
        maxSubjects = 11 #1 a 10
    elif dataset_name == 'bciciv2a':
        dataset = bciciv2a
        path = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/2a'
        maxSubjects = 10 #1 a 9

    resultDict = {
    'FBCSP' : {},
    'CSP' : {},
    }


    for key in resultDict.keys():
        for i in range(1,maxSubjects):
            classifiers = [lda(), 
                           nbpw(),
                           XGBClassifier(
                                n_estimators=300,
                                max_depth=4,
                                learning_rate=0.05,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                objective="binary:logistic",   # use "multi:softprob" se multiclasse
                                eval_metric="logloss",
                                random_state=42
                            )]
            for clf in classifiers:
                clf_name = clf.__class__.__name__
                if clf_name not in resultDict[key]:
                    resultDict[key][clf_name] = {}
                data = dataset(subject=i, path=path, labels=['left-hand', 'right-hand'])
                if key == 'FBCSP':
                        pre_folding = {'tf': (filterbank, {'kind_bp': 'chebyshevII'})}

                        sf = csp()
                        fe = logpower
                        fs = MIBIF(8, clf=lda())

                        pos_folding = {
                            'sf': (sf, {}),
                            'fe': (fe, {'flating': True}),
                            'fs': (fs, {}),
                            'clf': (clf, {})
                        }
                elif key == 'CSP':
                    pre_folding = {}
                    sf = csp()
                    fe = logpower
                    pos_folding = {
                        'fe': (fe, {'flating': True}),
                        'clf': (clf, {})
                    }
                ws =2.0
                start = time.time()
                results = kfold(
                    target=data,
                    start_window=data['events']['cue'][0] + 0.5,
                    window_size=2.0,
                    pre_folding=pre_folding,
                    pos_folding=pos_folding
                )
                end = time.time()
                print(f"Time taken for subject {i} with filter {key}: {end - start:.2f} seconds")
                filename = f"csvresults/{dataset_name}/{key}_{clf_name}_subject_{i}_start_{data['events']['cue'][0] + 0.5:.2f}_window_{ws:.2f}.csv".replace(".", "_")
                results.to_csv(filename, index=False,columns=['fold', 'tmin', 'true_label', *data["y_dict"].keys()])
                df = pd.DataFrame(results)
                metrics = classification_metrics(results)
                for metric_name in metrics.keys():
                    if metric_name != "confusion_matrix":
                        if metric_name not in resultDict[key][clf_name]:
                            # print(f"Initializing metric {metric_name} for filter {key} and classifier {clf_name}")
                            resultDict[key][clf_name][metric_name] = [metrics[metric_name]]
                        else:
                            resultDict[key][clf_name][metric_name].append(metrics[metric_name])
                            # print(f"{metric_name} for subject {i} with filter {key}, classifier {clf_name}: {metrics[metric_name]:.4f}")


                    else:
                        #plot confusion matrix
                        # print(f"Confusion Matrix for subject {i} with filter {key}:\n{metrics[metric_name]}")
                        plt.figure(figsize=(6, 5))
                        plt.imshow(metrics[metric_name], interpolation='nearest', cmap=plt.cm.Blues)
                        plt.title(f'Confusion Matrix for subject {i} with filter {key}')
                        plt.colorbar()
                        tick_marks = np.arange(len(data["y_dict"]))
                        plt.savefig(f'finalResults/{dataset_name}/confusion_matrix_{key}_subject_{i}_{clf_name}.png')
                        plt.close()



for key in resultDict.keys():
    for clfname in resultDict[key].keys():  
            final = pd.DataFrame(resultDict[key][clfname])
            final.to_csv(f"finalResults/{dataset_name}/{key}_{clfname}.csv", index=False)

# media das tabelonas para cada um dos results dos classifiers