import numpy as np
import pandas as pd
from pathlib import Path
from bciflow.datasets.cbcic import cbcic
from bciflow.datasets.bciciv2a import bciciv2a
from bciflow.datasets.bciciv2b import bciciv2b
from bciflow.modules.fe import logpower
# ============================================================
# Funções utilitárias
# ============================================================

def epoch_to_text(epoch, ch_names, max_points=30):
    """Converte uma época (n_channels, n_samples) em string curta para GPT."""
    parts = []
    for i, ch in enumerate(ch_names):
        sig = epoch[0,i]
        # Garante que seja 1D array de floats
        sig = np.asarray(sig).ravel()

        sig_str = ", ".join(f"{float(v):.3f}" for v in sig)
        parts.append(f"{ch}: [{sig_str}]")
    return "; ".join(parts)


def dataset_to_dataframe(dataset, max_points=30):
    """Converte o dicionário do dataset em DataFrame com colunas: features (texto), y_true."""
    X, y, ch_names, y_dict = dataset["X"], dataset["y"], dataset["ch_names"], dataset["y_dict"]
    rows = []
    for i in range(len(y)):
        feat_text = epoch_to_text(X[i], ch_names, max_points=max_points)
        rows.append({"features": feat_text, "y_true": y[i]})
    df = pd.DataFrame(rows)
    return df, y_dict

def build_label_defs(label_map):
    return "\n".join(f"- {k}: {v}" for k, v in label_map.items())

def build_examples(train_df, label_map,rng, n=2):

    # print(train_df)
    auxRows0 = train_df[train_df["y_true"] == 0]
    auxRows1 = train_df[train_df["y_true"] == 1]
   
    n = n//2
    seed0 = rng.integers(0, 1e9)
    seed1 = rng.integers(0, 1e9)

    rows0 = auxRows0.sample(min(n, len(auxRows0)), random_state=seed0)
    rows1 = auxRows1.sample(min(n, len(auxRows1)), random_state=seed1)
    
    #concat de rows0 e rows1
    rows = pd.concat([rows0, rows1])
    print(rows)
    chunks = []
    print("Building examples:")

    # garantir equilibrio de classes
    for _, r in rows.iterrows():
        feats = r["features"]
        label = r["y_true"]

        # print("feats:", feats)
        print("label:", label)
        #pegar a chave relacionada ao valor de label
        labelTraduct = None
        for key in label_map.keys():
            if label_map[key] == label:
                labelTraduct = key
                break
        # chunks.append(f"Entrada: {feats}\nSaída: {label} ({labelTraduct})")
        chunks.append(f"Exemplo de Entrada: {feats}\n Label dessa entrada: {labelTraduct}")
    return "\n\n".join(chunks)

# ============================================================
# Geração de prompts
# ============================================================


def generate_prompts_with_answers(dataset_name, dataset, subject_id, output_dir="prompts_out"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df, y_dict = dataset_to_dataframe(dataset)
    label_defs = build_label_defs(y_dict)
    valid_labels = ", ".join(str(l) for l in sorted(y_dict.keys()))

    # Split simples (80/20)
    df = df.sample(frac=1.0, random_state=42)
    n_train = int(0.8 * len(df))
    df_train, df_test = df.iloc[:n_train], df.iloc[n_train:]

    # Zero-shot prompts + respostas corretas
    zero_prompts = []
    zero_answers = []
    for _, row in df_test.iterrows():
        feats = row["features"]
        y_true = row["y_true"]
        prompt = f"""Você é um classificador de EEG.
Rótulos possíveis:
{label_defs}

Entrada:
{feats}

Responda apenas com a probabilidade de cada rótulo: {valid_labels}"""
        zero_prompts.append(prompt)
        labelTraduct = None
        for key, val in dataset["y_dict"].items():
            if val == y_true:
                labelTraduct = key
        zero_answers.append(labelTraduct)

    # Few-shot prompts + respostas corretas
    
    few_prompts = []
    few_answers = []
    rng = np.random.default_rng(seed=42)
    for _, row in df_test.iterrows():
        examples_text = build_examples(df_train, y_dict,rng, n=4)
        feats = row["features"]
        y_true = row["y_true"]
        prompt = f"""Você é um classificador de EEG.
Rótulos possíveis:
{label_defs}
Para cada entrada, retorne apenas as probabilidades dos rótulos, no formato exato: 

left-hand: <valor>, right-hand: <valor> 

Regras importantes: 

Use o rótulo verdadeiro para determinar qual classe deve ter maior probabilidade. 
Probabilidades devem somar 1. 
Distribua a probabilidade de forma plausível: a classe correta entre 0.6 e 0.9, a outra é o complemento (1 - probabilidade da correta).
Não retorne o rótulo direto, apenas o formato de probabilidades.

Exemplos de Entrada com o resultado exato da Saída:
{examples_text}

Agora classifique a entrada:
{feats}

Responda apenas com a probabilidade de cada rótulo: {valid_labels}"""
        few_prompts.append(prompt)
        labelTraduct = None
        for key, val in dataset["y_dict"].items():
            if val == y_true:
                labelTraduct = key
        few_answers.append(labelTraduct)

    # Caminhos dos arquivos
    zero_path = Path(output_dir) / f"{dataset_name}_subject{subject_id}_zero_shot_prompts.txt"
    few_path = Path(output_dir) / f"{dataset_name}_subject{subject_id}_few_shot_prompts.txt"

    zero_path_ans = Path(output_dir) / f"{dataset_name}_subject{subject_id}_zero_shot_answers.txt"
    few_path_ans = Path(output_dir) / f"{dataset_name}_subject{subject_id}_few_shot_answers.txt"

    # Salvar prompts + respostas
    with open(zero_path, "w", encoding="utf-8") as f:
        for p in zero_prompts:
            f.write(p + "\n\n---\n\n")
    with open(few_path, "w", encoding="utf-8") as f:
        for p in few_prompts:
            f.write(p + "\n\n---\n\n")

    # Salvar apenas respostas (um por linha)
    with open(zero_path_ans, "w", encoding="utf-8") as f:
        for ans in zero_answers:
            f.write(str(ans) + "\n")

    with open(few_path_ans, "w", encoding="utf-8") as f:
        for ans in few_answers:
            f.write(str(ans) + "\n")

    print(f"Arquivos salvos em:\n{zero_path}\n{few_path}\n{zero_path_ans}\n{few_path_ans}")
    return zero_prompts, zero_answers, few_prompts, few_answers


# ============================================================
# Loop principal
# ============================================================

for dataset_name in ['cbcic', 'bciciv2a', 'bciciv2b']:
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

    for i in range(1, 2):
        eegdata = dataset(subject=i, path=path, labels=['left-hand', 'right-hand'])
        startWind = int((eegdata['events']['cue'][0] + 0.5)*eegdata["sfreq"])
        stopWind = int(startWind + 1*eegdata["sfreq"])
        max =  int(eegdata["X"].shape[0] * 1)
        eegdata["X"] = eegdata["X"][:max, :,:, startWind:stopWind]
        eegdata["y"] = eegdata["y"][:max]
        eegdata["X"] = logpower(eegdata)["X"]
        print(eegdata["X"].shape)
        
        generate_prompts_with_answers(dataset_name, eegdata, subject_id=i)

        # Supondo que seu array seja chamado data com shape (288,1,22,25)
        # e que você tenha um vetor target com shape (288,)
        # Exemplo:
        # data = np.random.randn(288,1,22,25)
        # target = np.random.randint(0,2,288)

        # Remover a dimensão 'band' (já que é 1)
        data = eegdata["X"].reshape(eegdata["X"].shape[0],eegdata["X"].shape[-1])  # Agora shape = (288,22)
        print(data.shape)
        # Criar um DataFrame
        rows = {}
        rows.items()
        for trial_idx in range(data.shape[0]):
            for electrode_idx in range(data.shape[1]):
                if eegdata["ch_names"][electrode_idx] not in rows:
                    rows[eegdata["ch_names"][electrode_idx]] = []
                rows[eegdata["ch_names"][electrode_idx]].append(data[trial_idx, electrode_idx])
            if "true_label" not in rows:
                rows["true_label"] = []
            for  key,val in eegdata["y_dict"].items():
                if val == eegdata["y"][trial_idx]:
                    rows["true_label"].append(key)
        df = pd.DataFrame(rows)


        #separa em test e treino 80 20
        
        df.to_csv(f"prompts_out/csvs/data_{dataset_name}_subject_{i}.csv", index=False)
        
        split_index = int(0.8 * len(df))
        train_rows = df[:split_index]
        test_rows = df[split_index:]
        # Salvar em CSV

        for idx in range(split_index, len(df)):
           df.at[idx, "true_label"] = "nada"
        df.to_csv(f"prompts_out/csvs/data_{dataset_name}_subject_{i}_few.csv", index=False)
        # Salvar o novo arquivo final
        df.to_csv(f"prompts_out/csvs/final_data_{dataset_name}_subject_{i}.csv", index=False)
        df = pd.DataFrame(train_rows)
        df.to_csv(f"prompts_out/csvs/train_data_{dataset_name}_subject_{i}.csv", index=False)
        df = pd.DataFrame(test_rows).drop(columns=['true_label'])
        df.to_csv(f"prompts_out/csvs/test_data_{dataset_name}_subject_{i}.csv", index=False)

        



