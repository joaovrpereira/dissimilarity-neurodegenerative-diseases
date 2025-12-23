import pandas as pd
import os
import re

# --- Configuração ---
BASE_IMG_DIR = '/mnt/HD/oasis-dataset/'
CSV_DATA_PATH = 'OASIS3_dx1_simplified.csv'
TOLERANCIA_DIAS = 365  # Aceita diagnósticos até 1 ano antes ou depois da imagem

def get_all_mri_scans(base_dir):
    """
    Retorna TODAS as imagens T1w encontradas (sem filtrar a última ainda).
    """
    print(f"Mapeando todos os arquivos .nii.gz em '{base_dir}'...")
    if not os.path.isdir(base_dir):
        print(f"ERRO: Diretório '{base_dir}' não encontrado.")
        return pd.DataFrame()
        
    mri_scans = []
    # Regex para capturar OASISID, dia e o tipo de scan
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
    
    df_mri = df_mri.sort_values('image_path') # Garante consistência
    df_mri = df_mri.drop_duplicates(subset=['OASISID', 'days_to_visit'], keep='first')
    
    print(f"{len(df_mri)} sessões de MRI T1w encontradas (histórico completo).")
    return df_mri

def get_all_clinical_records(csv_path):
    """
    Retorna TODO o histórico clínico (sem filtrar o último ainda).
    """
    print(f"Carregando histórico clínico completo de '{csv_path}'...")
    df = pd.read_csv(csv_path)
    
    df['days_to_visit'] = pd.to_numeric(df['days_to_visit'], errors='coerce')
    df.dropna(subset=['days_to_visit'], inplace=True)
    df['days_to_visit'] = df['days_to_visit'].astype(int)
    
    if 'dx1' in df.columns:
        df.dropna(subset=['dx1'], inplace=True)

    df['days_clinical_ref'] = df['days_to_visit']

    print(f"{len(df)} registros clínicos carregados.")
    return df

def match_and_filter_latest(df_clinical, df_mri, tolerance=365):
    """
    1. Cruza todos os históricos.
    2. Filtra apenas o PAR VÁLIDO mais recente de cada paciente.
    """
    print("\n--- Iniciando Pareamento Temporal ---")
    
    df_clinical = df_clinical.sort_values('days_to_visit')
    df_mri = df_mri.sort_values('days_to_visit')

    merged = pd.merge_asof(
        df_mri, 
        df_clinical, 
        on='days_to_visit', 
        by='OASISID', 
        direction='nearest',
        tolerance=tolerance,
        suffixes=('_mri', '_clin')
    )
    
    valid_matches = merged.dropna(subset=['dx1']).copy()
    
    valid_matches['diff_dias'] = abs(valid_matches['days_to_visit'] - valid_matches['days_clinical_ref'])
    
    print(f"Total de pares (Imagem + Diagnóstico) válidos encontrados: {len(valid_matches)}")
    
    valid_matches.sort_values(by=['OASISID', 'days_to_visit'], inplace=True)
    
    latest_valid_df = valid_matches.drop_duplicates(subset='OASISID', keep='last')
    
    return latest_valid_df

if __name__ == '__main__':
    df_imagens = get_all_mri_scans(BASE_IMG_DIR)
    df_clinico = get_all_clinical_records(CSV_DATA_PATH)
    
    if not df_imagens.empty and not df_clinico.empty:
                
        df_final = match_and_filter_latest(df_clinico, df_imagens, tolerance=TOLERANCIA_DIAS)
        
        print("\n--- RESULTADO FINAL (Snapshot mais recente VALIDADO) ---")
        cols = ['OASISID', 'days_to_visit', 'image_path', 'dx1', 'diff_dias']
        print(df_final[cols].head())
        print(f"\nTotal de pacientes finais: {len(df_final)}")
        print(f"Média de diferença temporal: {df_final['diff_dias'].mean():.1f} dias")
        
        # Salva para o treino
        df_final.to_csv('dataset_multimodal_final.csv', index=False)
        print("Arquivo 'dataset_multimodal_final.csv' salvo com sucesso.")
        
    else:
        print("Erro: Dados insuficientes para processamento.")