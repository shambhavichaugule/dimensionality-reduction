import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os
from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io

# Load MNIST
dataset = load_dataset("ylecun/mnist")
train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()

# Convert images to pixel arrays
def image_to_pixels(image_dict):
    img = Image.open(io.BytesIO(image_dict['bytes']))
    return np.array(img).flatten()

print("Converting images...")
X_train = np.array([image_to_pixels(img) for img in train_df['image']])
y_train = train_df['label'].values

# Normalize
X_train = X_train / 255.0

print(f"Dataset shape: {X_train.shape}")
print(f"Each image: {X_train.shape[1]} features (28x28 pixels)")
print(f"Labels: {np.unique(y_train)}")

# ── PCA ──────────────────────────────────────────────────
print("\n" + "="*50)
print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("="*50)

# Step 1 — How many components do we need?
# Fit PCA with all components first
pca_full = PCA()
pca_full.fit(X_train)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find how many components explain 95% of variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1

print(f"Components needed to explain 95% variance: {n_components_95}")
print(f"Components needed to explain 99% variance: {n_components_99}")
print(f"Original features: 784")
print(f"Reduction ratio (95%): {784/n_components_95:.1f}x fewer features")

# Plot explained variance
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(cumulative_variance)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.axhline(y=0.99, color='g', linestyle='--', label='99% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA — Explained Variance')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(pca_full.explained_variance_ratio_[:50])
plt.xlabel('Component Number')
plt.ylabel('Explained Variance Ratio')
plt.title('Variance per Component (first 50)')

plt.tight_layout()
plt.savefig("pca_variance.png")
print("\nVariance chart saved!")

# Step 2 — Apply PCA with 154 components
pca = PCA(n_components=154)
X_train_pca = pca.fit_transform(X_train)

print(f"\nOriginal shape:  {X_train.shape}")
print(f"Reduced shape:   {X_train_pca.shape}")
print(f"Variance retained: {sum(pca.explained_variance_ratio_):.4f}")

# Step 3 — Visualize in 2D using first 2 components
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_train)

# Plot first 5000 points for visibility
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_2d[:5000, 0],
    X_2d[:5000, 1],
    c=y_train[:5000],
    cmap='tab10',
    alpha=0.5,
    s=1
)
plt.colorbar(scatter, label='Digit')
plt.title('PCA — MNIST in 2D (first 2 components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig("pca_2d.png")
print("2D visualization saved!")

# Step 4 — Reconstruct images to show information loss
pca_reconstruct = PCA(n_components=154)
X_compressed = pca_reconstruct.fit_transform(X_train)
X_reconstructed = pca_reconstruct.inverse_transform(X_compressed)

# Show original vs reconstructed
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(X_train[i].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=10)
axes[1, 0].set_ylabel('Reconstructed', fontsize=10)
plt.suptitle('PCA Reconstruction — Original vs 154 Components')
plt.savefig("pca_reconstruction.png")
print("Reconstruction chart saved!")

# Compare model accuracy with and without PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import io
import time

# Load test data
def image_to_pixels(image_dict):
    img = Image.open(io.BytesIO(image_dict['bytes']))
    return np.array(img).flatten()

test_df = dataset['test'].to_pandas()
X_test = np.array([image_to_pixels(img) for img in test_df['image']])
y_test = test_df['label'].values
X_test = X_test / 255.0

# Transform test data using already fitted PCA
X_test_pca = pca.transform(X_test)

# Train on subset for speed
X_sample = X_train[:10000]
y_sample = y_train[:10000]
X_sample_pca = X_train_pca[:10000]

# Model 1 — full 784 features
print("Training on full features...")
start = time.time()
lr_full = LogisticRegression(max_iter=1000)
lr_full.fit(X_sample, y_sample)
time_full = time.time() - start
acc_full = accuracy_score(y_test, lr_full.predict(X_test))

# Model 2 — 154 PCA components
print("Training on PCA features...")
start = time.time()
lr_pca = LogisticRegression(max_iter=1000)
lr_pca.fit(X_sample_pca, y_sample)
time_pca = time.time() - start
acc_pca = accuracy_score(y_test, lr_pca.predict(X_test_pca))

print(f"\n{'Model':<20} {'Features':>10} {'Accuracy':>10} {'Train Time':>12}")
print("-" * 55)
print(f"{'Full Features':<20} {'784':>10} {acc_full:>10.4f} {time_full:>10.2f}s")
print(f"{'PCA (95%)':<20} {'154':>10} {acc_pca:>10.4f} {time_pca:>10.2f}s")

# ── ICA ──────────────────────────────────────────────────
print("\n" + "="*50)
print("INDEPENDENT COMPONENT ANALYSIS (ICA)")
print("="*50)

# ICA works best on smaller number of components
# Use 50 components for speed
n_components_ica = 50

ica = FastICA(n_components=n_components_ica, random_state=42, max_iter=200)
X_train_ica = ica.fit_transform(X_train[:10000])  # subset for speed

print(f"Original shape:  {X_train[:10000].shape}")
print(f"ICA shape:       {X_train_ica.shape}")

# Visualize ICA components — what patterns did ICA find?
components = ica.components_

fig, axes = plt.subplots(5, 10, figsize=(15, 8))
axes = axes.flatten()
for i in range(50):
    axes[i].imshow(components[i].reshape(28, 28), cmap='gray')
    axes[i].axis('off')

plt.suptitle('ICA Components — Independent Patterns Found')
plt.tight_layout()
plt.savefig("ica_components.png")
print("ICA components saved!")

# Compare ICA vs PCA on classification
X_test_ica = ica.transform(X_test[:2000])
X_train_ica_sample = ica.transform(X_train[:10000])

lr_ica = LogisticRegression(max_iter=1000)
lr_ica.fit(X_train_ica_sample, y_train[:10000])
acc_ica = accuracy_score(y_test[:2000], lr_ica.predict(X_test_ica))

print(f"\nICA ({n_components_ica} components) accuracy: {acc_ica:.4f}")
print(f"PCA (154 components) accuracy: {acc_pca:.4f}")
print(f"Full features accuracy: {acc_full:.4f}")

# ── AUTOENCODER ──────────────────────────────────────────
print("\n" + "="*50)
print("AUTOENCODER")
print("="*50)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check if GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Convert data to PyTorch tensors
X_tensor = torch.FloatTensor(X_train[:10000]).to(device)
X_test_tensor = torch.FloatTensor(X_test[:2000]).to(device)

# Create dataloader
dataset_torch = TensorDataset(X_tensor, X_tensor)  # input = output
dataloader = DataLoader(dataset_torch, batch_size=256, shuffle=True)

# Define Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder — compress 784 → 32
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )

        # Decoder — reconstruct 32 → 784
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()  # output between 0 and 1
        )

    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)

# Train autoencoder
model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\nTraining Autoencoder...")
epochs = 20
losses = []

for epoch in range(epochs):
    epoch_loss = 0
    for batch_x, _ in dataloader:
        optimizer.zero_grad()
        reconstructed = model(batch_x)
        loss = criterion(reconstructed, batch_x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

# Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Loss')
plt.title('Autoencoder Training Loss')
plt.savefig("autoencoder_loss.png")
print("\nLoss chart saved!")

# Get compressed representations
model.eval()
with torch.no_grad():
    X_train_ae = model.encode(X_tensor).cpu().numpy()
    X_test_ae = model.encode(X_test_tensor).cpu().numpy()

print(f"\nOriginal shape:  {X_train[:10000].shape}")
print(f"Autoencoder shape: {X_train_ae.shape}")

# Visualize reconstructions
with torch.no_grad():
    sample = X_tensor[:10]
    reconstructed = model(sample).cpu().numpy()

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(X_train[i].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=10)
axes[1, 0].set_ylabel('Reconstructed', fontsize=10)
plt.suptitle('Autoencoder Reconstruction — Original vs 32 Components')
plt.savefig("autoencoder_reconstruction.png")
print("Reconstruction chart saved!")

# Classification with autoencoder features
lr_ae = LogisticRegression(max_iter=1000)
lr_ae.fit(X_train_ae, y_train[:10000])
acc_ae = accuracy_score(y_test[:2000],
         lr_ae.predict(X_test_ae))

print(f"\nAutoencoder (32 components) accuracy: {acc_ae:.4f}")
print(f"ICA         (50 components) accuracy: {acc_ica:.4f}")
print(f"PCA        (154 components) accuracy: {acc_pca:.4f}")
print(f"Full features               accuracy: {acc_full:.4f}")

# Final comparison visualization
methods = ['Full\nFeatures', 'PCA\n(154)', 'Autoencoder\n(32)', 'ICA\n(50)']
accuracies = [acc_full, acc_pca, acc_ae, acc_ica]
compressions = [1, 5.1, 24.5, 15.7]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy comparison
bars1 = ax1.bar(methods, accuracies, color=['gray', 'blue', 'green', 'orange'])
ax1.set_ylim(0.8, 0.95)
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy Comparison')
for bar, acc in zip(bars1, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=9)

# Compression comparison
bars2 = ax2.bar(methods, compressions, color=['gray', 'blue', 'green', 'orange'])
ax2.set_ylabel('Compression Ratio')
ax2.set_title('Compression Ratio (higher = fewer features)')
for bar, comp in zip(bars2, compressions):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{comp}x', ha='center', va='bottom', fontsize=9)

plt.suptitle('Dimensionality Reduction — Accuracy vs Compression')
plt.tight_layout()
plt.savefig("final_comparison.png")
print("Final comparison chart saved!")