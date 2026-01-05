import pandas as pd
from pathlib import Path

def build_dataframe(
    dataset_name,
    key,
    subject,
    window_size,
    cue_starts,
    fewOrZero="few",
    base_dir="prompts_out",
):
    results = []

    # Ler arquivo de respostas (true labels)
    answers_file = Path(base_dir) / f"{dataset_name}_subject{subject}_{fewOrZero}_shot_answers.txt"
    if not answers_file.exists():
        raise FileNotFoundError(f"Arquivo de respostas não encontrado: {answers_file}")
    true_labels = [line.strip() for line in open(answers_file, "r") if line.strip()]
    print(true_labels)
    RealStart = None
    
    for cue in cue_starts:
        start = cue
        preds_per_clf = []

    
        filename = (
            f"{dataset_name}_{key}_{fewOrZero}_subject_{subject}_start_{start:.2f}_window_{window_size:.2f}.csv"
        ).replace(".", "_")
        filepath = Path(base_dir) / filename

        if not filepath.exists():
            continue
        else:
            RealStart = start
        df = pd.read_csv(filepath)
        print(df)
        # garantir colunas esperadas
        expected_cols = ["left-hand", "right-hand"]
        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"Coluna {col} não encontrada em {filepath}")

        # adicionar colunas extras
        df["true_label"] = true_labels[: len(df)]  # alinhar tamanho
        df["tmin"] = start
        df["fold"] = 1
        expected_order = ["fold", "tmin", "true_label", "left-hand", "right-hand"]

        # Reorganizar colunas
        df_final = df[expected_order]

        print(df_final.head())

        results.append(df)

    return df_final, RealStart

dataset_names = ["cbcic", "bciciv2a"]
fewOrZero_opts = ["few", "zero"]
subjects = [1]  # exemplo de sujeitos
window_size = 1.0
cue_starts = [3.5, 2.5]
key = "GPT5"
base_dir = "prompts_out"
base_output_path = "csvresults/simplifiedResults/"
for dataset_name in dataset_names:
    for fewOrZero in fewOrZero_opts:
        for subject in subjects:
            df_final, rstart= build_dataframe(
                dataset_name=dataset_name,
                key=key,
                subject=subject,
                window_size=window_size,
                cue_starts=cue_starts,
                fewOrZero=fewOrZero,
                base_dir=base_dir,
            )

            if not df_final.empty:
                # salvar cada resultado separado
                filename = (
                f"{dataset_name}_{key}_{fewOrZero}_subject_{subject}_start_{rstart:.2f}_window_{window_size:.2f}.csv"
                ).replace(".", "_")
                output_file = Path(base_output_path) / filename
                df_final.to_csv(output_file, index=False)
                print(f" Resultado salvo em {output_file}")
            else:
                print(f" Nenhum resultado encontrado para {dataset_name}, {fewOrZero}, subject {subject}")



