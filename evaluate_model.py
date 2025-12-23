import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityd, Resized, EnsureTyped
from monai.data import DataLoader, Dataset
from monai.networks.nets import DenseNet121

# --- CONFIGURAÇÕES ---
IMG_SIZE = (96, 96, 96)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = 'dataset_multimodal_final.csv'

# 1. Recarregar os dados de Validação (O mesmo split do treino)
print("Preparando dados para validação...")
df = pd.read_csv(CSV_PATH)

def get_label(dx):
    dx = str(dx).lower()
    if "cognitively normal" in dx: return 0
    elif "ad dementia" in dx: return 1
    else: return -1

df['label'] = df['dx1'].apply(get_label)
df_clean = df[df['label'] != -1].copy()

from sklearn.model_selection import train_test_split
_, val_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean['label'])

val_files = [{"image": row['image_path'], "label": row['label']} for _, row in val_df.iterrows()]

val_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=IMG_SIZE),
    EnsureTyped(keys=["image", "label"]),
])

# Usamos Dataset simples aqui (não precisa de Cache para inferência rápida)
val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

# 2. Carregar o Modelo Salvo
print("Carregando o melhor modelo salvo...")
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(DEVICE)
model.load_state_dict(torch.load("melhor_modelo_alzheimer.pth"))
model.eval()

# 3. Gerar Predições
y_true = []
y_pred = []

print("Gerando predições (isso pode levar alguns segundos)...")
with torch.no_grad():
    for batch in val_loader:
        inputs = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        outputs = model(inputs)
        predictions = outputs.argmax(dim=1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

# 4. Relatórios
print("\n" + "="*40)
print("RELATÓRIO DE CLASSIFICAÇÃO")
print("="*40)
print(classification_report(y_true, y_pred, target_names=['Controle (0)', 'Alzheimer (1)']))

# 5. Matriz de Confusão Visual
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Controle', 'Alzheimer'], yticklabels=['Controle', 'Alzheimer'])
plt.ylabel('Real')
plt.xlabel('Predito')
plt.title('Matriz de Confusão - Validação')
plt.savefig('matriz_confusao.png')
print("\nMatriz de confusão salva como 'matriz_confusao.png'")
print(f"Total de amostras validadas: {len(y_true)}")