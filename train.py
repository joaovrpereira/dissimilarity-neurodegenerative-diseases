import pandas as pd
import torch
import numpy as np
import monai
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityd, Resized, EnsureTyped, RandRotate90d, RandFlipd
)
from monai.networks.nets import DenseNet121
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# --- CONFIGURAÇÕES ---
CSV_PATH = 'dataset_multimodal_final.csv'
IMG_SIZE = (96, 96, 96) 
BATCH_SIZE = 4          
EPOCHS = 30             
LR = 1e-5              
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Usando dispositivo: {DEVICE}")

print("Carregando CSV e preparando labels...")
df = pd.read_csv(CSV_PATH)

def get_label(dx):
    dx = str(dx).lower()
    if "cognitively normal" in dx:
        return 0  # Saudável
    elif "ad dementia" in dx:
        return 1  # Alzheimer
    else:
        return -1 
    
df['label'] = df['dx1'].apply(get_label)

# Filtrar apenas classes 0 e 1
df_clean = df[df['label'] != -1].copy()
print(f"Total de pacientes para treino/teste: {len(df_clean)}")
print(df_clean['label'].value_counts())

# Divisão Treino (80%) / Validação (20%)
train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean['label'])

# Criar lista de dicionários para o MONAI
train_files = [{"image": row['image_path'], "label": row['label']} for _, row in train_df.iterrows()]
val_files = [{"image": row['image_path'], "label": row['label']} for _, row in val_df.iterrows()]


train_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")), # Padroniza tamanho do voxel
    ScaleIntensityd(keys=["image"]), # Normaliza intensidade
    Resized(keys=["image"], spatial_size=IMG_SIZE), # Redimensiona para 96x96x96
    RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 2)),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=IMG_SIZE),
    EnsureTyped(keys=["image", "label"]),
])

print("Criando Datasets (Isso pode demorar um pouco para carregar na RAM)...")

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=2)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# DenseNet121 3D
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(DEVICE)
# --- CÁLCULO DOS PESOS (WEIGHTED LOSS) ---
class_weights = torch.tensor([1.0, 5.0]).to(DEVICE)

loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

print("\n--- INICIANDO TREINAMENTO ---")

for epoch in range(EPOCHS):
    print(f"-" * 10)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(DEVICE), batch_data["label"].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if step % 10 == 0:
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"Epoch {epoch + 1} finalizada. Média Loss: {epoch_loss:.4f}")

    # --- VALIDAÇÃO ---
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        for val_data in val_loader:
            val_images, val_labels = val_data["image"].to(DEVICE), val_data["label"].to(DEVICE)
            val_outputs = model(val_images)
            value = torch.eq(val_outputs.argmax(dim=1), val_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
        
        metric = num_correct / metric_count
        metric_values.append(metric)
        
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "melhor_modelo_alzheimer.pth")
            print(">>> Novo melhor modelo salvo!")
            
        print(f"Acurácia atual: {metric:.4f} | Melhor acurácia: {best_metric:.4f} na epoch {best_metric_epoch}")

print(f"Treinamento concluído. Melhor Acurácia: {best_metric:.4f}")