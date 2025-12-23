import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from monai.networks.nets import DenseNet121
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityd, Resized, EnsureTyped
)
from sklearn.model_selection import train_test_split

# --- CONFIGURAÇÕES ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = 'dataset_multimodal_final.csv'
MODEL_PATH = 'melhor_modelo_alzheimer.pth' 
IMG_SIZE = (96, 96, 96)
BATCH_SIZE = 8 
LIMIAR_DECISAO = 0.20  

print(f"--- GERANDO RESULTADOS FINAIS (ENSEMBLE) ---")
print(f"Dispositivo: {DEVICE}")
print("Carregando dataset...")
df = pd.read_csv(CSV_PATH)

def get_label(dx):
    dx = str(dx).lower()
    if "cognitively normal" in dx: return 0
    elif "ad dementia" in dx: return 1
    else: return -1

df['label'] = df['dx1'].apply(get_label)
df_clean = df[df['label'] != -1].copy()

# Split Estratificado
train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean['label'])

print(f"Total Pacientes: {len(df_clean)}")
print(f"Treino: {len(train_df)} | Validação: {len(val_df)}")

# Pipeline de Transformação
transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=IMG_SIZE),
    EnsureTyped(keys=["image", "label"]),
])

train_files = [{"image": row['image_path'], "label": row['label']} for _, row in train_df.iterrows()]
val_files = [{"image": row['image_path'], "label": row['label']} for _, row in val_df.iterrows()]

# Datasets
train_ds = CacheDataset(data=train_files, transform=transforms, cache_rate=1.0, num_workers=2)
val_ds = CacheDataset(data=val_files, transform=transforms, cache_rate=1.0, num_workers=2)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Carregando DenseNet de '{MODEL_PATH}'...")
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.class_layers.out = torch.nn.Identity() 
model.eval()

def extract_features(dataloader, name):
    print(f"Extraindo características do {name}...")
    features = []
    labels_list = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["image"].to(DEVICE)
            labels = batch["label"].cpu().numpy()
            emb = model(inputs)
            features.append(emb.cpu().numpy())
            labels_list.append(labels)
    return np.vstack(features), np.concatenate(labels_list)

X_train, y_train = extract_features(train_loader, "Treino")
X_val, y_val = extract_features(val_loader, "Validação")

print("\n--- Treinando Ensemble (SVM + Random Forest + Regressão Logística) ---")

clf1 = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced')
clf2 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf3 = LogisticRegression(class_weight='balanced', max_iter=1000)

ensemble = VotingClassifier(
    estimators=[('svm', clf1), ('rf', clf2), ('lr', clf3)],
    voting='soft' #Faz a média
)

ensemble.fit(X_train, y_train)

y_proba_ensemble = ensemble.predict_proba(X_val)[:, 1]

print("\n--- DEBUG DAS PROBABILIDADES DO ENSEMBLE ---")
print(f"Probabilidade Mínima: {y_proba_ensemble.min():.4f}")
print(f"Probabilidade Máxima: {y_proba_ensemble.max():.4f}")
print("-" * 50)
print(f"{'Limiar':<10} | {'TP (Alzheimer Detectado)':<25} | {'FP (Alarme Falso)':<20}")
print("-" * 50)

for limiar_teste in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    y_teste = (y_proba_ensemble >= limiar_teste).astype(int)
    cm_teste = confusion_matrix(y_val, y_teste)
    tp = cm_teste[1, 1] 
    fp = cm_teste[0, 1] 
    print(f"{limiar_teste:<10.2f} | {tp:<25} | {fp:<20}")

print("-" * 50)
print(f"Gerando imagens finais com LIMIAR ESCOLHIDO: {LIMIAR_DECISAO}")

y_pred_ajustado = (y_proba_ensemble >= LIMIAR_DECISAO).astype(int)

# --- RELATÓRIOS E IMAGENS ---

# Relatório
print("\n" + "="*50)
print("RELATÓRIO FINAL - ENSEMBLE")
print("="*50)
print(classification_report(y_val, y_pred_ajustado, target_names=['Controle', 'Alzheimer']))

# Matriz
cm = confusion_matrix(y_val, y_pred_ajustado)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Controle', 'Alzheimer'], 
            yticklabels=['Real: Controle', 'Real: Alzheimer'])
plt.title(f'Matriz Ensemble (Limiar {LIMIAR_DECISAO})')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.tight_layout()
plt.savefig('tcc_matriz_ensemble.png', dpi=300)
print(">>> Salva: tcc_matriz_ensemble.png")

# ROC
fpr, tpr, _ = roc_curve(y_val, y_proba_ensemble)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'Ensemble ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Performance do Ensemble (SVM+RF+LR)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('tcc_roc_ensemble.png', dpi=300)
print(">>> Salva: tcc_roc_ensemble.png")