import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
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

print(f"--- GERANDO VISUALIZAÇÃO COMPARATIVA DOS 3 MODELOS ---")

# 1. CARREGAR DADOS E MODELO (Igual antes)
print("Carregando e processando dados...")
df = pd.read_csv(CSV_PATH)
def get_label(dx):
    dx = str(dx).lower()
    if "cognitively normal" in dx: return 0
    elif "ad dementia" in dx: return 1
    else: return -1
df['label'] = df['dx1'].apply(get_label)
df_clean = df[df['label'] != -1].copy()

# Usamos o conjunto de validação para o plot ficar limpo
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

model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.class_layers.out = torch.nn.Identity()
model.eval()

# 2. EXTRAIR FEATURES
print("Extraindo deep features...")
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

# 3. PCA (Reduzir para 2D para podermos desenhar)
print("Aplicando PCA (1024 -> 2 dimensões)...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# --- FUNÇÃO DE PLOTAGEM ---
def plot_decision_boundary(clf, X, y, title, filename):
    print(f"Gerando gráfico para: {title}...")
    
    # Treina o modelo nos dados 2D (só para visualização)
    clf.fit(X, y)
    
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Criar grade de fundo
    xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    yy = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
    # Prever o fundo
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(xy).reshape(XX.shape)
        # Linha de decisão suave
        ax.contourf(XX, YY, Z, levels=[-100, 0, 100], alpha=0.2, colors=['#1f77b4', '#ff7f0e'])
        ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
    else:
        # Para Random Forest (que não tem decision_function linear)
        Z = clf.predict_proba(xy)[:, 1].reshape(XX.shape)
        ax.contourf(XX, YY, Z, levels=[0, 0.5, 1], alpha=0.2, colors=['#1f77b4', '#ff7f0e'])
        ax.contour(XX, YY, Z, colors='k', levels=[0.5], alpha=0.5, linestyles=['-'])

    # Plotar os pacientes
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette={0: '#1f77b4', 1: '#ff7f0e'}, s=100, edgecolor='k')
    
    plt.title(f'Fronteira de Decisão: {title}\n(Visualização PCA 2D)', fontsize=14)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    
    # Legenda
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) > 0:
        plt.legend(handles, ['Controle', 'Alzheimer'], title='Diagnóstico')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f">>> Salva: {filename}")

# --- 4. GERAR AS 3 IMAGENS ---

# 1. SVM (O Líder)
svm_viz = SVC(kernel="linear", C=1.0) # Linear fica mais fácil de ver no 2D
plot_decision_boundary(svm_viz, X_pca, y, "Support Vector Machine (SVM)", "viz_svm.png")

# 2. Random Forest (O Detalhista)
rf_viz = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
plot_decision_boundary(rf_viz, X_pca, y, "Random Forest", "viz_rf.png")

# 3. Regressão Logística (O Estatístico)
lr_viz = LogisticRegression(max_iter=1000)
plot_decision_boundary(lr_viz, X_pca, y, "Regressão Logística", "viz_lr.png")

print("\nConcluído! Verifique as imagens viz_svm.png, viz_rf.png e viz_lr.png")