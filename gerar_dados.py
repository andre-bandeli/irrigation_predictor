import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def generate_synthetic_data_bootstrap(df_original, num_new_rows, output_file_name='base_aumentada.csv'):

    print(f"Gerando {num_new_rows} novos dados")

    categorical_cols = df_original.select_dtypes(include='object').columns.tolist()

    df_temp = df_original.copy()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_temp[col] = le.fit_transform(df_temp[col])
        label_encoders[col] = le

    synthetic_data = df_temp.sample(n=num_new_rows, replace=True, random_state=42)
    synthetic_data = synthetic_data.reset_index(drop=True)

    for col in categorical_cols:
        synthetic_data[col] = label_encoders[col].inverse_transform(synthetic_data[col])

    numeric_cols = df_original.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if col != 'Farm_ID':
            std_dev = df_original[col].std()
            noise = np.random.normal(0, std_dev * 0.05, num_new_rows)
            synthetic_data[col] += noise
            synthetic_data[col] = synthetic_data[col].apply(lambda x: max(0, x))

    max_farm_id = df_original['Farm_ID'].max()
    if isinstance(max_farm_id, (int, float)):
        synthetic_data['Farm_ID'] = range(max_farm_id + 1, max_farm_id + 1 + num_new_rows)
    else:
        synthetic_data['Farm_ID'] = [f'Farm_SYN_{i}' for i in range(num_new_rows)]

    synthetic_data.to_csv(output_file_name, index=False)
    print(f"Dados salvos em '{output_file_name}'")
    print(f"Shape dos dados: {synthetic_data.shape}")

    return synthetic_data

if __name__ == "__main__":
    
    try:
        df_original = pd.read_csv('irrigacao.csv')
    except FileNotFoundError:
        print("Erro: 'irrigacao.csv' não encontrado.")
        exit()

    num_new_rows_to_generate = 500

    df_synthetic = generate_synthetic_data_bootstrap(df_original, num_new_rows_to_generate, 'planilha_nova.csv')

    df_combined = pd.concat([df_original, df_synthetic], ignore_index=True)
    print("\nShape do dataset combinado (original + novos dados):", df_combined.shape)
    print("Primeiras 5 linhas do dataset combinado:")
    print(df_combined.head())