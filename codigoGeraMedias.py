import pandas as pd
from pathlib import Path
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def str_to_array(s):
    lista = ast.literal_eval(s)   # transforma string em lista Python
    return np.array(lista)        # converte lista em numpy array


def save_metric_csv(dataset_name, method, metric, output_dir="finalResults/summary"):
    # Lê os dois arquivos
    filepath = Path(output_dir) / f"{dataset_name}_{method}_committee_metrics.csv"
    df1 = pd.read_csv(filepath)
    
    out_path = Path(output_dir) / f"{dataset_name}_{metric}.csv"
    df2 = pd.read_csv(out_path)
    
    classic_clfs = ["LinearDiscriminantAnalysis", "nbpw", "XGBClassifier","Comite"]
    
    # Dicionário final para armazenar resultados
    results = {}
    if "confusion_matrix" == metric:
        
        if type(df1[metric][0]) == str:
            df1[metric] = df1[metric].apply(str_to_array)
        plt.figure(figsize=(6, 5))
        mean_matrix = df1[metric].mean()
        sns.heatmap(mean_matrix,
                    annot=True, fmt=".0f", cmap="Blues",
                    xticklabels=["Mão Esquerda", "Mão Direita"],
                    yticklabels=["Mão Esquerda", "Mão Direita"])

        plt.title("Média das Matrizes (valores brutos)")
        plt.xlabel("Predição")
        plt.ylabel("Real")
        plt.tight_layout()
        # plt.colorbar(plt.cm.ScalarMappable(cmap="Blues"))  # adiciona barra de cores
        plt.savefig(f'finalResults/{dataset_name}/average_confusion_matrix_{method}_{classic_clfs[-1]}.png')
        plt.close()

        
    mean_committee = df1[metric].mean()
   
    results[classic_clfs[-1]] = mean_committee
    # Itera sobre colunas de df2 (métricas individuais)
    for key in df2.columns:
        splitVals = str(key).split("_")
        
        # Verifica se a coluna corresponde ao método desejado
        if splitVals[0] == method:
            clf_name = "_".join(splitVals[1:])  # nome do classificador
            
            # Média dos valores individuais
            if "confusion_matrix" == metric:
                if type(df2[key][0]) == str:
                    df2[key] = df2[key].apply(str_to_array)
                plt.figure(figsize=(6, 5))
                mean_matrix = df2[key].mean()
                sns.heatmap(mean_matrix,
                            annot=True, fmt=".0f", cmap="Blues",
                            xticklabels=["Mão Esquerda", "Mão Direita"],
                            yticklabels=["Mão Esquerda", "Mão Direita"])

                plt.title("Média das Matrizes (valores brutos)")
                plt.xlabel("Predição")
                plt.ylabel("Real")
                plt.tight_layout()
                # plt.colorbar(plt.cm.ScalarMappable(cmap="Blues"))  # adiciona barra de cores
                plt.savefig(f'finalResults/{dataset_name}/average_confusion_matrix_{key}_{clf_name}.png')
                plt.close()
            mean_individual = df2[key].mean()
            
            # Média do comitê (se existir no df1)
            # Monta dicionário com origem e valores
           
            results[clf_name] = mean_individual
            

    
    return results


# Exemplo de uso: percorre todas as métricas
summary_all = {}
for dataset in ["bciciv2a", "cbcic"]:
    summary_all[dataset] = {}
    for method in ["CSP", "FBCSP"]:
        summary_all[dataset][method] = {}
        for metric in ["accuracy", "precision", "recall", "f1", "auc","confusion_matrix"]:
            summary_all[dataset][method][metric] = save_metric_csv(dataset, method, metric)

import pandas as pd

def dict_to_tables(results_dict):
    tables = {}
    for dataset, methods in results_dict.items():
        for method, metrics in methods.items():
            # monta uma lista de linhas
            rows = {}
            for metric, classifiers in metrics.items():
                for clf, mean_val in classifiers.items():
                    if clf not in rows:
                        rows[clf] = {}
                    rows[clf][metric] = mean_val
            
            # cria DataFrame com classificadores como linhas e métricas como colunas
            df = pd.DataFrame.from_dict(rows, orient="index")
            df.index.name = "Classificador"
            df.to_csv(f"final_Table_{dataset}_{method}.csv")
            tables[(dataset, method)] = df
    return tables

# Exemplo de uso
tables = dict_to_tables(summary_all)


# # Visualizar uma tabela específica
# print("=== bciciv2a - CSP ===")
# # print(tables[("bciciv2a", "CSP")])
# if keyV == "confusion_matrix":
#             plt.figure(figsize=(6, 5))
#             plt.imshow(metrics[keyV], interpolation='nearest', cmap=plt.cm.Blues)
#             plt.colorbar()
#             plt.savefig(f'finalResults/{dataset_name}/confusion_matrix_{key}_average_comite.png')
#             plt.close()