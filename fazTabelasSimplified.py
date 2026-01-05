import pandas as pd
from pathlib import Path
from codes.metrics import classification_metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from codes.metrics import classification_metrics  # ajuste conforme seu projeto

def collect_metrics(
    dataset_name,
    keys,
    subjects,
    window_sizes,
    cue_starts,
    base_dir="csvresults/simplifiedResults/",
):
    """
    Retorna:
    { metodo: {classificador: {metric_name: [val_s1, val_s2, ...]} } }
    """
    # separa complementos
    classic_clfs = ["LinearDiscriminantAnalysis", "nbpw", "XGBClassifier"]
    gpt_modes = ["few", "zero"]

    # inicializa dicionário
    result_dict = {
        key: {
            clf: {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "auc": [],
                "confusion_matrix": []
            }
            for clf in classic_clfs
        }
        for key in keys if key != "GPT5"
    }

    result_dict["GPT5"] = {
        mode: {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auc": [],
            "confusion_matrix": []
        }
        for mode in gpt_modes
    }

    # loop principal
    for key in keys:
        if key == "GPT5":
            clf_list = gpt_modes
        else:
            clf_list = classic_clfs

        for clf_name in clf_list:
            for subject in subjects:
                subject_metrics = {m: [] for m in result_dict[key][clf_name].keys()}

                for ws in window_sizes:
                    for cue in cue_starts:
                        start = cue
                        filename = (
                            f"{dataset_name}_{key}_{clf_name}_subject_{subject}_start_{start:.2f}_window_{ws:.2f}.csv"
                        ).replace(".", "_")
                        filepath = Path(base_dir) / filename
                        

                        if not filepath.exists():
                            continue

                        df = pd.read_csv(filepath)
                        print("Checking file:", filepath)
                        metrics = classification_metrics(df)
                        
                        for m in subject_metrics.keys():
                            subject_metrics[m].append(metrics[m])

                # agrega por sujeito (média)
                for m in result_dict[key][clf_name].keys():
                    if len(subject_metrics[m]) == 0:
                        result_dict[key][clf_name][m].append(np.nan)
                    else:
                        if m == "confusion_matrix":
                            resul = np.array(subject_metrics[m][0]).tolist()
                            result_dict[key][clf_name][m].append(resul)
                        else:
                            result_dict[key][clf_name][m].append(subject_metrics[m][0])

    return result_dict

def committee_metrics(
    dataset_name,
    key,
    subject,
    window_sizes,
    cue_starts,
    classifiers=["LinearDiscriminantAnalysis", "nbpw", "XGBClassifier"],
    base_dir="csvresults/simplifiedResults/",
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
                            f"{dataset_name}_{key}_{clf_name}_subject_{subject}_start_{start:.2f}_window_{ws:.2f}.csv"
                ).replace(".", "_")
                filepath = Path(base_dir) / filename
                # print(filepath)
                
                if not filepath.exists():
                    continue
                df = pd.read_csv(filepath)
                # print(df.head())
                
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
        return {m: np.nan for m in ["accuracy", "precision", "recall", "f1", "auc","confusion_matrix"]}

    # concatena todas as predições
    df_committee = columns12.copy()

    committee_pred = np.concatenate(all_preds)
        # cria dataframe para passar na função de métricas adicionando as duas primeiras colunas do df original
    df_committee["true_label"] = y_true
    df_committee["left-hand"] = committee_pred[0]
    df_committee["right-hand"] = committee_pred[1]    
    metrics = classification_metrics(df_committee)

    finalmetrics = {}
    for keyV in metrics.keys():
        if keyV != "balanced_accuracy" and keyV != "mcc":
            if keyV == "confusion_matrix":
                finalmetrics[keyV] = metrics[keyV].tolist()
            else:
                finalmetrics[keyV] = metrics[keyV]
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

    out_path = Path(output_dir) / f"Simplified_{dataset_name}_{metric}.csv"
    df.to_csv(out_path)
    
    print(f"Salvo: {out_path}")
    return data


# =========================
# Execução
# =========================

datasets_cfg = {
    "bciciv2a": {"maxSubjects": 9},

    "cbcic": {"maxSubjects": 10},
}

keys = ["CSP", "FBCSP","GPT5"]
window_sizes = [1.0]
cue_starts = [2.5, 3.5]
AllDAtaframes = {}
for dataset_name, cfg in datasets_cfg.items():
    subjects = range(1, 2)
    if dataset_name not in AllDAtaframes:
        AllDAtaframes[dataset_name] = {}
    result_dict = collect_metrics(
        dataset_name=dataset_name,
        keys=keys,
        subjects=subjects,
        window_sizes=window_sizes,
        cue_starts=cue_starts,
    )

    for metric in ["accuracy", "precision", "recall", "f1", "auc","confusion_matrix"]:
        df = save_metric_csv(result_dict, dataset_name, metric)
        if metric not in AllDAtaframes[dataset_name]:
            AllDAtaframes[dataset_name][metric] = None
        AllDAtaframes[dataset_name][metric] = df
        # print(df.head())


comiteData = {}
keys = ["CSP", "FBCSP"]
for dataset_name, cfg in datasets_cfg.items():
    if dataset_name not in comiteData:
        comiteData[dataset_name] = {}
    subjects = range(1, 2)
    for key in keys:
        result_dict = []
        if key not in comiteData[dataset_name]:
            comiteData[dataset_name][key] = None
        
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
        comiteData[dataset_name][key] = result_dict
        df = pd.DataFrame(result_dict)
        df.to_csv(Path("finalResults/summary") / f"Simplified_{dataset_name}_{key}_committee_metrics.csv")

import pandas as pd

def unify_dicts(dict1, dict2):
    rows = []

    # --- Primeiro dicionário ---
    for dataset, metrics in dict1.items():
        for metric, classifiers in metrics.items():
            for clf_key, values in classifiers.items():
                # separa método e classificador
                parts = clf_key.split("_")
                if len(parts) == 2:
                    method, clf = parts
                else:
                    method, clf = parts[0], "_".join(parts[1:])
                val = values[0] if isinstance(values, list) else values
                rows.append({
                    "Dataset": dataset,
                    "Método": method,
                    "Classificador": clf,
                    "Métrica": metric,
                    "Valor": val
                })

    # --- Segundo dicionário ---
    for dataset, methods in dict2.items():
        for method, clf_list in methods.items():
            for clf_metrics in clf_list:  # lista de dicts
                for metric, val in clf_metrics.items():
                    rows.append({
                        "Dataset": dataset,
                        "Método": method,
                        "Classificador": "committee",
                        "Métrica": metric,
                        "Valor": val
                    })

    return pd.DataFrame(rows)

def make_tables_by_dataset(df):
    tables = {}
    for dataset, group in df.groupby("Dataset"):
        # cria coluna Modelo
        group = group.assign(Modelo=group["Método"] + "_" + group["Classificador"])
        
        # gera tabela pivot e salva
        pivot = group.pivot(index="Modelo", columns="Métrica", values="Valor")
        pivot.to_csv(f"Simplified_final_Table_{dataset}.csv")
        tables[dataset] = pivot

        # cria pasta de saída se não existir

        # para cada linha, plota apenas se for confusion_matrix
        for _, row in group.iterrows():
            if row["Métrica"] == "confusion_matrix":
                modelo = row["Modelo"]
                matriz = row["Valor"]   # assumindo que já é np.array 2x2

                plt.figure(figsize=(6, 5))
                sns.heatmap(matriz,
                            annot=True, fmt=".0f", cmap="Blues",
                            xticklabels=["Mão Esquerda", "Mão Direita"],
                            yticklabels=["Mão Esquerda", "Mão Direita"])
                
                plt.title(f"Matriz de Confusão - {modelo}")
                plt.xlabel("Predição")
                plt.ylabel("Real")
                plt.tight_layout()
                
                plt.savefig(f'finalResults/{dataset}/simplified_confusion_matrix_{modelo}.png')
                plt.close()
    
    return tables



df_unificado = unify_dicts(AllDAtaframes, comiteData)

tables = make_tables_by_dataset(df_unificado)


# Visualizar tabela de um dataset
print("=== bciciv2a ===")
print(tables["bciciv2a"])



