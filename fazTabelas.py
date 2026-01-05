import pandas as pd
from pathlib import Path
from codes.metrics import classification_metrics
import numpy as np

def collect_metrics(
    dataset_name,
    keys,
    subjects,
    window_sizes,
    cue_starts,
    base_dir="csvresults",
):
    """
    Retorna:
    { metodo: {classificador: {metric_name: [val_s1, val_s2, ...]} } }
    """
    classifiers = [
        "LinearDiscriminantAnalysis",
        "nbpw",
        "XGBClassifier",
    ]

    result_dict = {
        key: {
            clf: {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "auc": [],
            }
            for clf in classifiers
        }
        for key in keys
    }

    for key in keys:
        for clf_name in classifiers:
            for subject in subjects:
                subject_metrics = {m: [] for m in result_dict[key][clf_name].keys()}

                for ws in window_sizes:
                    for cue in cue_starts:
                        start = cue 

                        filename = (
                            f"{key}_{clf_name}_subject_{subject}_start_{start:.2f}_window_{ws:.2f}_csv"
                        ).replace(".", "_")
                        filepath = Path(base_dir) / dataset_name / filename
                        # print("Checking file:", filepath)
                        if not filepath.exists():
                            continue
                        df = pd.read_csv(filepath)
                       
                        metrics = classification_metrics(df)

                        for m in subject_metrics.keys():
                            subject_metrics[m].append(metrics[m])
                    # print(f"Subject{subject}: {subject_metrics}")
                # agrega por sujeito (média)
                for m in result_dict[key][clf_name].keys():
                    if len(subject_metrics[m]) == 0:
                        result_dict[key][clf_name][m].append(np.nan)
                    else:
                        result_dict[key][clf_name][m].append(np.mean(subject_metrics[m]))

    return result_dict


def committee_metrics(
    dataset_name,
    key,
    subject,
    window_sizes,
    cue_starts,
    classifiers=["LinearDiscriminantAnalysis", "nbpw", "XGBClassifier"],
    base_dir="csvresults",
):
    """
    Calcula métricas do comitê (ensemble) dos classificadores.
    Retorna: {metric_name: valor}
    """
    all_preds = []
    y_true = None
    df = None
    for ws in window_sizes:
        for cue in cue_starts:
            start = cue
            
            preds_per_clf = []
            for clf_name in classifiers:
                filename = (
                            f"{key}_{clf_name}_subject_{subject}_start_{start:.2f}_window_{ws:.2f}_csv"
                        ).replace(".", "_")
                filepath = Path(base_dir) / dataset_name / filename
                print(filepath)
                if not filepath.exists():
                    continue

                df = pd.read_csv(filepath)
                print(df.head())
               
                # assumindo que df tem colunas: y_true, y_pred
                if y_true is None:
                    y_true = df["true_label"].values

                preds_per_clf.append(df["left-hand"].values)

            if len(preds_per_clf) == len(classifiers):
                # Votação majoritária
                preds_per_clf = np.array(preds_per_clf)  # shape: (n_classifiers, n_samples)
                committee_pred = np.mean(preds_per_clf, axis=0)  # média -> voto
                committee_pred2  = 1 - committee_pred 
                all_preds.append((committee_pred,committee_pred2))
            
    columns12 = df.iloc[:, :2]   # todas as linhas, apenas as 2 primeiras colunas


    if not all_preds or y_true is None:
        return {m: np.nan for m in ["accuracy", "precision", "recall", "f1", "auc"]}

    # concatena todas as predições
    df_committee = columns12.copy()

    committee_pred = np.concatenate(all_preds)
        # cria dataframe para passar na função de métricas adicionando as duas primeiras colunas do df original
    df_committee["true_label"] = y_true
    df_committee["left-hand"] = committee_pred[0]
    df_committee["right-hand"] = committee_pred[1]    
    metrics = classification_metrics(df_committee)

    finalmetrics = {}
    for key in metrics.keys():
        if key != "confusion_matrix" and key != "balanced_accuracy" and key != "mcc":
            finalmetrics[key] = metrics[key]
    return finalmetrics


# =========================
# Salvamento
# =========================

def save_metric_csv(result_dict, dataset_name, metric, output_dir="finalResults/summary"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Flatten para DataFrame
    data = {}
    for key in result_dict.keys():
        for clf in result_dict[key].keys():
            col_name = f"{key}_{clf}"
            data[col_name] = result_dict[key][clf][metric]

    df = pd.DataFrame(data)
    df.index.name = "subject"

    out_path = Path(output_dir) / f"{dataset_name}_{metric}.csv"
    df.to_csv(out_path)

    print(f"Salvo: {out_path}")
    return df


# =========================
# Execução
# =========================

datasets_cfg = {
    "bciciv2a": {"maxSubjects": 9},
    "bciciv2b": {"maxSubjects": 9},
    "cbcic": {"maxSubjects": 10},
}

keys = ["CSP", "FBCSP"]
window_sizes = [2.0]
cue_starts = [2.5, 3.5]

for dataset_name, cfg in datasets_cfg.items():
    subjects = range(1, cfg["maxSubjects"] + 1)

    result_dict = collect_metrics(
        dataset_name=dataset_name,
        keys=keys,
        subjects=subjects,
        window_sizes=window_sizes,
        cue_starts=cue_starts,
    )

    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        df = save_metric_csv(result_dict, dataset_name, metric)
        # print(df.head())


for dataset_name, cfg in datasets_cfg.items():
    subjects = range(1, cfg["maxSubjects"] + 1)
    for key in keys:
        result_dict = []
        for subject in subjects:
            metrics = committee_metrics(
                dataset_name=dataset_name,
                key=key,
                subject=subject,
                window_sizes=window_sizes,
                cue_starts=cue_starts,
            )
            # print(f"{dataset_name} - {key} - subject {subject}: {metrics}")
            result_dict.append(metrics)  # remove AUC para evitar conflito de chaves
        df = pd.DataFrame(result_dict)
        df.to_csv(Path("finalResults/summary") / f"{dataset_name}_{key}_committee_metrics.csv")

    