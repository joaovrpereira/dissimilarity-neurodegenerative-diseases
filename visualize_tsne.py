import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE  # <--- A ESTRELA DO SHOW
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
PERPLEXITY = 30 # Parâmetro do t-SNE (pode testar 5, 30, 50)

print(f"--- GERANDO VISUALIZAÇÃO t-SNE (SOTA para Embeddings) ---")

# 1. CARREGAR DADOS
df = pd.read_csv(CSV_PATH)
def get_label(dx):
    dx = str(dx).lower()
    if "cognitively normal" in dx: return 0
    elif "ad dementia" in dx: return 1
    else: return -1
df['label'] = df['dx1'].apply(get_label)
df_clean = df[df['label'] != -1].copy()

# Usar validação para visualização limpa
train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean['label'])

transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=IMG_SIZE),
    EnsureTyped(keys=["image", "label"]),
])

files = [{"image": row['image_path'], "label": row['label']} for _, row in val_df.iterrows()]
ds = CacheDataset(data=files, transform=transforms, cache_rate=1.0, num_workers=2)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 2. CARREGAR MODELO
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.class_layers.out = torch.nn.Identity()
model.eval()

# 3. EXTRAIR FEATURES
print("Extraindo features...")
features = []
labels_list = []
with torch.no_grad():
    for batch in loader:
        inputs = batch["image"].to(DEVICE)
        labels = batch["label"].cpu().numpy()
        emb = model(inputs)
        features.append(emb.cpu().numpy())
        labels_list.append(labels)

X = np.vstack(features)
y = np.concatenate(labels_list)

# 4. APLICAR t-SNE
print("Calculando t-SNE (Isso pode levar alguns segundos)...")
# t-SNE reduz de 1024 para 2 dimensões focando em agrupar vizinhos
tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=42, init='pca', learning_rate='auto')
X_embedded = tsne.fit_transform(X)

# 5. PLOTAR
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, 
                palette={0: '#1f77b4', 1: '#ff7f0e'}, s=100, alpha=0.8, edgecolor='k')

plt.title(f'Visualização t-SNE das Características Latentes\n(Separação de Clusters)', fontsize=14)
plt.xlabel('Dimensão t-SNE 1')
plt.ylabel('Dimensão t-SNE 2')

# Legenda
ax = plt.gca()
handles, _ = ax.get_legend_handles_labels()
plt.legend(handles, ['Controle', 'Alzheimer'], title='Diagnóstico')

plt.tight_layout()
plt.savefig('pacientes_tsne.png', dpi=300)
print(">>> Imagem salva: pacientes_tsne.png")