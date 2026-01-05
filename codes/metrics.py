import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef
)

def classification_metrics(results: pd.DataFrame, average="macro"):
    """
    Calcula métricas completas a partir de um DataFrame contendo
    true_label e probabilidades por classe.

    Parâmetros
    ----------
    results : pd.DataFrame
        DataFrame com:
        - coluna 'true_label'
        - colunas de probabilidade a partir da coluna 3

    average : str
        'binary', 'macro', 'micro', 'weighted'

    Retorna
    -------
    dict
        Dicionário com métricas e matriz de confusão
    """

    y_true = results["true_label"].values
    probs  = results.iloc[:, 3:].values
    y_pred = results.iloc[:, 3:].idxmax(axis=1).values
    metrics = {}

    metrics["accuracy"]  = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=average)
    metrics["recall"]    = recall_score(y_true, y_pred, average=average)
    metrics["f1"]        = f1_score(y_true, y_pred, average=average)

    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # AUC
    if probs.shape[1] == 2:
        # binário
        metrics["auc"] = roc_auc_score(y_true, probs[:, 1])
    else:
        # multiclasse
        metrics["auc"] = roc_auc_score(
            y_true,
            probs,
            multi_class="ovr",
            average=average
        )

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    return metrics
