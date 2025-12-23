import pandas as pd
import os
import re

# --- Configuração ---
BASE_IMG_DIR = '/mnt/HD/oasis-dataset/'
CSV_DATA_PATH = 'OASIS3_dx1_simplified.csv'
TOLERANCIA_DIAS = 365 

def get_all_mri_scans(base_dir):
    print(f"Mapeando todos os arquivos .nii.gz em '{base_dir}'...")
    if not os.path.isdir(base_dir):
        print(f"ERRO: Diretório '{base_dir}' não encontrado.")
        return pd.DataFrame()
        
    mri_scans = []
    bids_pattern = re.compile(r'sub-(OAS3\d+)_ses-d(\d+)_.*_(\w+)\.nii\.gz')

    for root, _, files in os.walk(base_dir):
        for file in files:
            if 'T1w.nii.gz' in file:
                match = bids_pattern.search(file)
                if match:
                    oasis_id, days, scan_type = match.groups()
                    if scan_type == 'T1w':
                        mri_scans.append({
                            'OASISID': oasis_id,
                            'days_to_visit': int(days),
                            'image_path': os.path.join(root, file)
                        })
    
    if not mri_scans:
        return pd.DataFrame()

    df_mri = pd.DataFrame(mri_scans)
    df_mri = df_mri.drop_duplicates(subset=['OASISID', 'days_to_visit'], keep='first')
    
    print(f"{len(df_mri)} imagens T1w encontradas.")
    return df_mri

def get_all_clinical_records(csv_path):
    print(f"Carregando dados clínicos de '{csv_path}'...")
    df = pd.read_csv(csv_path)
    
    df['days_to_visit'] = pd.to_numeric(df['days_to_visit'], errors='coerce')
    df.dropna(subset=['days_to_visit'], inplace=True)
    df['days_to_visit'] = df['days_to_visit'].astype(int)
    
    if 'dx1' in df.columns:
        df.dropna(subset=['dx1'], inplace=True)

    print(f"{len(df)} registros clínicos carregados.")
    return df

def match_temporal_data(df_clinical, df_mri, tolerance=365):
    """
    Realiza o pareamento temporal inteligente (Merge AsOf).
    """
    print("\n--- Iniciando Pareamento Temporal (Nearest Match) ---")
    
    # 1. PREPARAÇÃO (CORREÇÃO AQUI):
    # Criamos uma cópia explícita da coluna de dias clínicos.
    # Assim, ela sobreviverá ao merge com um nome exclusivo.
    df_clinical = df_clinical.copy()
    df_clinical['days_clinical_ref'] = df_clinical['days_to_visit']
    
    # 2. Ordenação OBRIGATÓRIA
    df_clinical = df_clinical.sort_values('days_to_visit')
    df_mri = df_mri.sort_values('days_to_visit')
    
    # 3. Merge AsOf
    merged = pd.merge_asof(
        df_mri, 
        df_clinical, 
        on='days_to_visit', 
        by='OASISID', 
        direction='nearest',
        tolerance=tolerance,
        suffixes=('_mri', '_clin') # Suffixes só afetam colunas duplicadas que NÃO são a chave
    )
    
    # 4. Remove quem não encontrou par
    valid_matches = merged.dropna(subset=['dx1']).copy()
    
    # 5. Cálculo da diferença (USANDO A COLUNA NOVA)
    # Agora usamos 'days_to_visit' (original da MRI) e 'days_clinical_ref' (cópia do clínico)
    valid_matches['diff_dias'] = abs(valid_matches['days_to_visit'] - valid_matches['days_clinical_ref'])
    
    print(f"Total de pares válidos encontrados: {len(valid_matches)}")
    print(f"Média de diferença temporal: {valid_matches['diff_dias'].mean():.1f} dias")
    
    return valid_matches

if __name__ == '__main__':
    df_imagens = get_all_mri_scans(BASE_IMG_DIR)
    df_clinico = get_all_clinical_records(CSV_DATA_PATH)
    
    if not df_imagens.empty and not df_clinico.empty:
        
        df_pareado = match_temporal_data(df_clinico, df_imagens, tolerance=TOLERANCIA_DIAS)
        
        # Filtra snapshot mais recente
        df_final = df_pareado.sort_values('days_to_visit', ascending=True)
        df_final = df_final.drop_duplicates(subset='OASISID', keep='last')
        
        print("\n--- RESULTADO FINAL ---")
        # Ajustei o print para mostrar a coluna correta
        cols_to_show = ['OASISID', 'days_to_visit', 'days_clinical_ref', 'diff_dias', 'dx1']
        print(df_final[cols_to_show].head())
        print(f"\nTotal de pacientes finais: {len(df_final)}")
        
        # Opcional: Salvar CSV
        df_final.to_csv('dataset_multimodal_final.csv', index=False)
        
    else:
        print("Erro: Dados insuficientes.")