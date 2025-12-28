import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import rasterio
from rasterio.transform import from_bounds
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, accuracy_score, precision_score, \
    recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from tqdm import tqdm
import warnings
import json
from collections import defaultdict, Counter
import random
import time
import datetime

warnings.filterwarnings('ignore')


# Set random seeds to ensure repeatability
def set_seed(seed=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(1000)


# ==============================
# Configuration Parameters
# ==============================
class Config:
    # Data paths - using pre-extracted balanced dataset
    EXCEL_PATH = r"E:\balanced_samples.xlsx"  # The input file includes features arranged in chronological order, with the last column being the sample label, and each line corresponding to one sample

    FEATURE_DIRS = [
        r"E:\time1",
        r"E:\time2",
        r"E:\time3",
        r"E:\time4",
        r"E:\time5",
        r"E:\time6",
        r"E:\time7",
        r"E:\time8",
        r"E:\time9",
        r"E:\time10",
        r"E:\time11",
        r"E:\time12",
    ]

    # Data parameters
    TEMPORAL_DIM = 12
    FEATURE_DIM = 25
    NUM_CLASSES = 18

    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0002
    NUM_EPOCHS = 100
    PATIENCE = 15
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DROPOUT = 0.1

    # Ablation experiment settings
    NUM_REPEATS = 10  # Number of repetitions for each experiment
    TRAIN_RATIO = 0.8

    # Output paths
    OUTPUT_DIR = "output"
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
    RESULT_SAVE_DIR = os.path.join(OUTPUT_DIR, "results")
    CLASSIFICATION_SAVE_DIR = os.path.join(OUTPUT_DIR, "classification_maps")
    TIME_SAVE_DIR = os.path.join(OUTPUT_DIR, "time_stats")
    ATTENTION_SAVE_DIR = os.path.join(OUTPUT_DIR, "attention_weights")


# Create output directories
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(Config.RESULT_SAVE_DIR, exist_ok=True)
os.makedirs(Config.CLASSIFICATION_SAVE_DIR, exist_ok=True)
os.makedirs(Config.TIME_SAVE_DIR, exist_ok=True)
os.makedirs(Config.ATTENTION_SAVE_DIR, exist_ok=True)


# ==============================
# Multi-scale Temporal Attention Network (MSTA-Net) - Modified Version
# ==============================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=5, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Single-layer TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.temporal_projection = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        transformer_out = self.transformer(x)
        temporal_weights = self.temporal_projection(transformer_out)
        temporal_weights = F.softmax(temporal_weights, dim=1)
        attended_output = transformer_out * temporal_weights

        # Save attention weights for visualization
        self.temporal_attention_weights = temporal_weights.squeeze(-1).detach().cpu().numpy()

        return attended_output, temporal_weights.squeeze(-1)

    def get_temporal_attention_weights(self):
        return self.temporal_attention_weights if hasattr(self, 'temporal_attention_weights') else None


class FeatureChannelAttention(nn.Module):
    def __init__(self, temporal_dim, num_heads=3, dropout=0.1):
        super().__init__()
        self.temporal_dim = temporal_dim

        # Single-layer TransformerEncoder with d_model=12
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=num_heads,
            dim_feedforward=temporal_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.feature_projection = nn.Linear(temporal_dim, 1)

    def forward(self, x):
        # Transpose input: (batch_size, temporal_dim, feature_dim) -> (batch_size, feature_dim, temporal_dim)
        x_transposed = x.transpose(1, 2)

        transformer_out = self.transformer(x_transposed)
        feature_weights = self.feature_projection(transformer_out)
        feature_weights = F.softmax(feature_weights, dim=1)
        attended_output = transformer_out * feature_weights

        # Transpose back to original dimensions
        attended_output = attended_output.transpose(1, 2)

        # Save attention weights for visualization
        self.feature_attention_weights = feature_weights.squeeze(-1).detach().cpu().numpy()

        return attended_output, feature_weights.squeeze(-1)

    def get_feature_attention_weights(self):
        return self.feature_attention_weights if hasattr(self, 'feature_attention_weights') else None


class MultiScaleTemporalAttentionNet(nn.Module):
    def __init__(self, temporal_dim=12, feature_dim=25, num_classes=18, dropout=0.3):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Temporal attention - keep feature dimension as 25
        self.temporal_attention = TemporalAttention(hidden_dim=feature_dim, num_heads=5, dropout=dropout)

        # Feature channel attention - use temporal dimension 12
        self.feature_attention = FeatureChannelAttention(temporal_dim=temporal_dim, num_heads=3, dropout=dropout)

        # Multi-scale temporal convolution - input 25, output 16 for each branch
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(feature_dim, 16, kernel_size=3, padding=1),
            nn.Conv1d(feature_dim, 16, kernel_size=5, padding=2),
            nn.Conv1d(feature_dim, 16, kernel_size=7, padding=3),
            nn.Conv1d(feature_dim, 16, kernel_size=1),
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim * 114, 512),  # 12 * (25 + 25 + 64) = 12 * 114
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.temporal_dim, self.feature_dim)

        # Temporal attention
        temporal_attended, temporal_weights = self.temporal_attention(x)

        # Feature channel attention
        feature_attended, feature_weights = self.feature_attention(x)

        # Multi-scale temporal feature extraction
        temporal_features = []
        x_transposed = x.transpose(1, 2)  # (batch_size, feature_dim, temporal_dim)

        for conv in self.temporal_convs:
            conv_out = conv(x_transposed)
            conv_out = F.relu(conv_out)
            # Keep temporal dimension, no pooling
            temporal_features.append(conv_out)

        # Concatenate multi-scale features along feature dimension
        multi_scale_temporal = torch.cat(temporal_features, dim=1)  # (batch_size, 64, temporal_dim)
        multi_scale_temporal = multi_scale_temporal.transpose(1, 2)  # (batch_size, temporal_dim, 64)

        # Concatenate outputs from three modules
        combined_features = torch.cat([temporal_attended, feature_attended, multi_scale_temporal], dim=2)

        # Flatten
        combined_flat = combined_features.reshape(batch_size, -1)

        # Classification
        output = self.classifier(combined_flat)

        attention_outputs = {
            'temporal_weights': temporal_weights,
            'feature_weights': feature_weights,
            'temporal_attention_matrix': self.temporal_attention.get_temporal_attention_weights(),
            'feature_attention_matrix': self.feature_attention.get_feature_attention_weights()
        }

        return output, attention_outputs


# ==============================
# Data Loading and Processing
# ==============================
class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_balanced_dataset(config):
    """Load pre-balanced dataset directly"""
    print("Loading balanced dataset...")
    df = pd.read_excel(config.EXCEL_PATH)

    # Separate features and labels
    feature_columns = [col for col in df.columns if col != 'label']
    features = df[feature_columns].values
    labels = df['label'].values

    print(f"Balanced data: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Label distribution: {Counter(labels)}")

    return features, labels


def split_dataset(features, labels, train_ratio=0.8):
    """Stratified dataset splitting by class"""
    train_features, train_labels = [], []
    val_features, val_labels = [], []

    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)

        split_idx = int(len(label_indices) * train_ratio)

        train_indices = label_indices[:split_idx]
        val_indices = label_indices[split_idx:]

        train_features.append(features[train_indices])
        train_labels.append(labels[train_indices])
        val_features.append(features[val_indices])
        val_labels.append(labels[val_indices])

    train_features = np.vstack(train_features)
    train_labels = np.hstack(train_labels)
    val_features = np.vstack(val_features)
    val_labels = np.hstack(val_labels)

    # Shuffle
    train_indices = np.random.permutation(len(train_features))
    val_indices = np.random.permutation(len(val_features))

    return train_features[train_indices], train_labels[train_indices], val_features[val_indices], val_labels[
        val_indices]


# ==============================
# Model Training and Evaluation
# ==============================
def train_model(model, train_loader, val_loader, config, experiment_id):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Record training start time
    train_start_time = time.time()
    epoch_times = []

    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()

        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(config.DEVICE)
            batch_labels = batch_labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs, _ = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(config.DEVICE)
                batch_labels = batch_labels.to(config.DEVICE)

                outputs, _ = model(batch_features)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss_avg)

        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(config.MODEL_SAVE_DIR, f'best_model_exp_{experiment_id}.pth'))
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch + 1}/{config.NUM_EPOCHS}: '
                  f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%, '
                  f'Epoch Time: {epoch_time:.2f}s')

    # Calculate total training time
    total_train_time = time.time() - train_start_time

    model.load_state_dict(
        torch.load(os.path.join(config.MODEL_SAVE_DIR, f'best_model_exp_{experiment_id}.pth')))

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'total_train_time': total_train_time,
        'epoch_times': epoch_times,
        'actual_epochs': len(epoch_times)
    }


def evaluate_model(model, data_loader, config):
    model.eval()
    all_predictions = []
    all_labels = []
    all_temporal_attention = []
    all_feature_attention = []

    # Record evaluation start time
    eval_start_time = time.time()

    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(config.DEVICE)
            outputs, attention_outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

            # Save attention weights
            if attention_outputs['temporal_attention_matrix'] is not None:
                all_temporal_attention.append(attention_outputs['temporal_attention_matrix'])
            if attention_outputs['feature_attention_matrix'] is not None:
                all_feature_attention.append(attention_outputs['feature_attention_matrix'])

    # Calculate evaluation time
    eval_time = time.time() - eval_start_time

    accuracy = accuracy_score(all_labels, all_predictions)
    kappa = cohen_kappa_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    cm = confusion_matrix(all_labels, all_predictions)

    return {
        'accuracy': accuracy,
        'kappa': kappa,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'eval_time': eval_time,
        'temporal_attention': all_temporal_attention,
        'feature_attention': all_feature_attention
    }


# ==============================
# Full Region Classification Functionality
# ==============================
def load_region_features(config):
    """Load full region feature data - ensure processing of all invalid values"""
    print("Loading full region feature data...")

    # Record loading start time
    load_start_time = time.time()

    first_time_dir = config.FEATURE_DIRS[0]
    feature_files = sorted(glob.glob(os.path.join(first_time_dir, "*.tif")))

    if len(feature_files) != config.FEATURE_DIM:
        print(f"Warning: Expected {config.FEATURE_DIM} features, but found {len(feature_files)} files")

    # Read first feature file to get geographic information
    with rasterio.open(feature_files[0]) as src:
        height, width = src.shape
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        bounds = src.bounds  # Get boundary information

    print(f"Original data geographic information:")
    print(f"  Coordinate system: {crs}")
    print(f"  Transformation matrix: {transform}")
    print(f"  Bounds: {bounds}")
    print(f"  Dimensions: {width} x {height}")

    # Initialize full region feature array
    region_features = np.zeros((height, width, config.TEMPORAL_DIM, config.FEATURE_DIM), dtype=np.float32)

    # Load data by time period and feature
    for t, time_dir in enumerate(tqdm(config.FEATURE_DIRS, desc="Loading time periods")):
        feature_files = sorted(glob.glob(os.path.join(time_dir, "*.tif")))

        for f, feature_file in enumerate(tqdm(feature_files, desc=f"Time period {t + 1}", leave=False)):
            if f >= config.FEATURE_DIM:
                break

            with rasterio.open(feature_file) as src:
                data = src.read(1)

                # Thoroughly process invalid values
                if src.nodata is not None:
                    # Replace NoData with 0
                    data = np.where(data == src.nodata, 0, data)

                # Handle all possible invalid values
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                # Ensure no negative values (if features should be non-negative)
                data = np.maximum(data, 0)

                region_features[:, :, t, f] = data

    print(f"Full region feature data shape: {region_features.shape}")

    # Check data quality
    total_pixels = height * width
    valid_pixels = np.sum(np.any(region_features != 0, axis=(2, 3)))  # Pixels with at least one non-zero feature
    print(f"Data quality: {valid_pixels}/{total_pixels} pixels have valid data ({valid_pixels / total_pixels * 100:.2f}%)")

    # Calculate loading time
    load_time = time.time() - load_start_time
    print(f"Full region data loading time: {load_time:.2f} seconds")

    return region_features, height, width, transform, crs, profile, bounds, load_time


def predict_region(model, region_features, config, batch_size=1000):
    """Perform full region classification prediction - ensure all pixels have prediction results"""
    print("Performing full region classification prediction...")

    # Record prediction start time
    predict_start_time = time.time()

    height, width = region_features.shape[0], region_features.shape[1]
    region_flat = region_features.reshape(-1, config.TEMPORAL_DIM * config.FEATURE_DIM)

    # Batch prediction to avoid memory overflow
    predictions = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(region_flat), batch_size), desc="Prediction progress"):
            batch_data = region_flat[i:i + batch_size]
            batch_tensor = torch.FloatTensor(batch_data).to(config.DEVICE)

            outputs, _ = model(batch_tensor)
            _, batch_pred = torch.max(outputs.data, 1)

            predictions.extend(batch_pred.cpu().numpy())

    # Reshape to image shape
    prediction_map = np.array(predictions).reshape(height, width)

    # Statistics of prediction results
    unique_classes, counts = np.unique(prediction_map, return_counts=True)
    print("Predicted class distribution:")
    for cls, count in zip(unique_classes, counts):
        print(f"  Class {cls}: {count} pixels ({count / (height * width) * 100:.2f}%)")

    # Calculate prediction time
    predict_time = time.time() - predict_start_time
    print(f"Full region prediction time: {predict_time:.2f} seconds")

    return prediction_map, predict_time


def save_prediction_map(prediction_map, transform, crs, profile, bounds, output_path):
    """Save prediction results as TIFF file - maintain identical coordinate system as original feature files"""
    # Record save start time
    save_start_time = time.time()

    # Use original profile as base to ensure completely consistent coordinate system
    output_profile = profile.copy()

    # Only update necessary parameters
    output_profile.update({
        'dtype': rasterio.uint8,
        'count': 1,
        'compress': 'lzw',  # Keep compression
        'nodata': None,  # Don't set NoData, 0 is a valid class
    })

    # Ensure correct data type
    prediction_map = prediction_map.astype(np.uint8)

    # Write file - use original profile to ensure complete coordinate consistency
    with rasterio.open(output_path, 'w', **output_profile) as dst:
        dst.write(prediction_map, 1)

        # Add descriptive label
        dst.set_band_description(1, "MSTA-Net Classification Result")

    # Calculate save time
    save_time = time.time() - save_start_time

    print(f"Classification results saved to: {output_path}")
    print(f"Prediction map information:")
    print(f"  Shape: {prediction_map.shape}")
    print(f"  Data type: {prediction_map.dtype}")
    print(f"  Class range: {np.min(prediction_map)} to {np.max(prediction_map)}")
    print(f"  Save time: {save_time:.2f} seconds")

    # Verify output file coordinates match original file exactly
    with rasterio.open(output_path) as src:
        print(f"Output file coordinate verification:")
        print(f"  Coordinate system: {src.crs}")
        print(f"  Transformation matrix: {src.transform}")
        print(f"  Bounds: {src.bounds}")
        print(f"  Dimensions: {src.width} x {src.height}")

        # Check if coordinates match original file
        if src.crs == crs:
            print("  ✓ Coordinate system consistent")
        else:
            print("  ✗ Coordinate system inconsistent!")

        if np.allclose(np.array(src.transform), np.array(transform)):
            print("  ✓ Transformation matrix consistent")
        else:
            print("  ✗ Transformation matrix inconsistent!")

        if np.allclose(np.array(src.bounds), np.array(bounds)):
            print("  ✓ Bounds consistent")
        else:
            print("  ✗ Bounds inconsistent!")

    return save_time


# ==============================
# Attention Weight Visualization
# ==============================
def plot_attention_weights(temporal_attention, feature_attention, save_path):
    """Plot attention weight heatmaps"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Temporal attention heatmap
    if len(temporal_attention) > 0:
        temporal_avg = np.mean(np.vstack(temporal_attention), axis=0)
        sns.heatmap(temporal_avg.reshape(1, -1), annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=[f'T{i + 1}' for i in range(len(temporal_avg))],
                    yticklabels=['Attention'], ax=ax1)
        ax1.set_title('Temporal Attention Weights')

    # Feature attention heatmap
    if len(feature_attention) > 0:
        feature_avg = np.mean(np.vstack(feature_attention), axis=0)
        sns.heatmap(feature_avg.reshape(1, -1), annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=[f'F{i + 1}' for i in range(len(feature_avg))],
                    yticklabels=['Attention'], ax=ax2)
        ax2.set_title('Feature Attention Weights')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save attention weight data separately
    if len(temporal_attention) > 0:
        np.save(save_path.replace('.png', '_temporal_attention.npy'), np.vstack(temporal_attention))
    if len(feature_attention) > 0:
        np.save(save_path.replace('.png', '_feature_attention.npy'), np.vstack(feature_attention))


# ==============================
# Result Visualization and Saving
# ==============================
def plot_confusion_matrix(cm, class_names, save_path, normalize=True):
    """Plot and save confusion matrix - add percentage display"""

    if normalize:
        # Calculate confusion matrix in percentage form
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_percent = np.round(cm_percent, 1)

    # Create two subplots: numerical form and percentage form
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    # Numerical confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # Percentage confusion matrix
    if normalize:
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Percentage %)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Also save separate percentage confusion matrix
    if normalize:
        percent_save_path = save_path.replace('.png', '_percentage.png')
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix (Percentage %)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(percent_save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history['train_losses'], label='Training Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_accuracies'], label='Training Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results_to_excel(results, time_stats, output_path):
    """Save experimental results to Excel"""

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Create detailed results table
        metrics_data = []
        for i, result in enumerate(results):
            metrics_data.append({
                'Experiment': i + 1,
                'Accuracy': result['accuracy'],
                'Kappa': result['kappa'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1_Score': result['f1_score'],
                'Train_Time(s)': result['train_time'],
                'Eval_Time(s)': result['eval_time']
            })

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel(writer, sheet_name='experiment_results', index=False)

        # Save statistical information
        stats_data = {
            'Metric': ['Accuracy', 'Kappa', 'Precision', 'Recall', 'F1_Score', 'Train_Time(s)', 'Eval_Time(s)'],
            'Mean': [
                metrics_df['Accuracy'].mean(),
                metrics_df['Kappa'].mean(),
                metrics_df['Precision'].mean(),
                metrics_df['Recall'].mean(),
                metrics_df['F1_Score'].mean(),
                metrics_df['Train_Time(s)'].mean(),
                metrics_df['Eval_Time(s)'].mean()
            ],
            'Std': [
                metrics_df['Accuracy'].std(),
                metrics_df['Kappa'].std(),
                metrics_df['Precision'].std(),
                metrics_df['Recall'].std(),
                metrics_df['F1_Score'].std(),
                metrics_df['Train_Time(s)'].std(),
                metrics_df['Eval_Time(s)'].std()
            ],
            'Max': [
                metrics_df['Accuracy'].max(),
                metrics_df['Kappa'].max(),
                metrics_df['Precision'].max(),
                metrics_df['Recall'].max(),
                metrics_df['F1_Score'].max(),
                metrics_df['Train_Time(s)'].max(),
                metrics_df['Eval_Time(s)'].max()
            ],
            'Min': [
                metrics_df['Accuracy'].min(),
                metrics_df['Kappa'].min(),
                metrics_df['Precision'].min(),
                metrics_df['Recall'].min(),
                metrics_df['F1_Score'].min(),
                metrics_df['Train_Time(s)'].min(),
                metrics_df['Eval_Time(s)'].min()
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='experiment_stats', index=False)

        # Save best confusion matrix
        best_exp_idx = np.argmax([r['accuracy'] for r in results])
        best_cm = results[best_exp_idx]['confusion_matrix']
        cm_df = pd.DataFrame(best_cm)
        cm_df.to_excel(writer, sheet_name='best_confusion_matrix', index=False)

        # Save percentage confusion matrix
        cm_percent = best_cm.astype('float') / best_cm.sum(axis=1)[:, np.newaxis] * 100
        cm_percent_df = pd.DataFrame(cm_percent)
        cm_percent_df.to_excel(writer, sheet_name='best_confusion_matrix_pct', index=False)

        # Save time statistics table
        time_data = [{
            'Model': 'MSTA-Net',
            'Avg_Train_Time(s)': np.mean(time_stats['train_times']),
            'Std_Train_Time(s)': np.std(time_stats['train_times']),
            'Avg_Eval_Time(s)': np.mean(time_stats['eval_times']),
            'Std_Eval_Time(s)': np.std(time_stats['eval_times']),
            'Region_Predict_Time(s)': time_stats['region_classification_time']['predict_time'],
            'Region_Save_Time(s)': time_stats['region_classification_time']['save_time'],
            'Data_Load_Time(s)': time_stats['region_classification_time']['load_time'],
            'Total_Region_Time(s)': (time_stats['region_classification_time']['predict_time'] +
                                     time_stats['region_classification_time']['save_time'] +
                                     time_stats['region_classification_time']['load_time'])
        }]

        time_df = pd.DataFrame(time_data)
        time_df.to_excel(writer, sheet_name='time_statistics', index=False)

    print(f"Experimental results saved to: {output_path}")


def save_time_statistics_to_csv(time_stats, output_path):
    """Save time statistics separately to CSV file"""
    time_data = [{
        'Model': 'MSTA-Net',
        'Avg_Train_Time(s)': np.mean(time_stats['train_times']),
        'Std_Train_Time(s)': np.std(time_stats['train_times']),
        'Min_Train_Time(s)': np.min(time_stats['train_times']),
        'Max_Train_Time(s)': np.max(time_stats['train_times']),
        'Avg_Eval_Time(s)': np.mean(time_stats['eval_times']),
        'Std_Eval_Time(s)': np.std(time_stats['eval_times']),
        'Min_Eval_Time(s)': np.min(time_stats['eval_times']),
        'Max_Eval_Time(s)': np.max(time_stats['eval_times']),
        'Region_Predict_Time(s)': time_stats['region_classification_time']['predict_time'],
        'Region_Save_Time(s)': time_stats['region_classification_time']['save_time'],
        'Data_Load_Time(s)': time_stats['region_classification_time']['load_time'],
        'Total_Region_Time(s)': (time_stats['region_classification_time']['predict_time'] +
                                 time_stats['region_classification_time']['save_time'] +
                                 time_stats['region_classification_time']['load_time']),
        'Total_Avg_Time(s)': (np.mean(time_stats['train_times']) + np.mean(time_stats['eval_times']) +
                              time_stats['region_classification_time']['predict_time'] +
                              time_stats['region_classification_time']['save_time'] +
                              time_stats['region_classification_time']['load_time'])
    }]

    time_df = pd.DataFrame(time_data)
    time_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Time statistics saved to: {output_path}")


# ==============================
# Main Execution Flow
# ==============================
def main():
    print("=" * 60)
    print("Multi-scale Temporal Attention Network (MSTA-Net) - Modified Model")
    print("=" * 60)

    config = Config()

    # Record overall start time
    overall_start_time = time.time()

    # Load data
    features, labels = load_balanced_dataset(config)

    # Experimental results storage
    all_results = []
    train_times = []
    eval_times = []
    best_accuracy = 0.0
    best_exp_id = 0

    # Repeat experiments
    for exp_id in range(config.NUM_REPEATS):
        print(f"\n--- Experiment {exp_id + 1}/{config.NUM_REPEATS} ---")

        set_seed(1000 + exp_id)

        # Split dataset
        train_features, train_labels, val_features, val_labels = split_dataset(
            features, labels, config.TRAIN_RATIO
        )

        # Create data loaders
        train_dataset = TimeSeriesDataset(train_features, train_labels)
        val_dataset = TimeSeriesDataset(val_features, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        print(f"Training set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")

        # Initialize modified model
        model = MultiScaleTemporalAttentionNet(
            temporal_dim=config.TEMPORAL_DIM,
            feature_dim=config.FEATURE_DIM,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        ).to(config.DEVICE)

        # Train model
        trained_model, history = train_model(
            model, train_loader, val_loader, config, exp_id
        )

        # Evaluate model
        eval_results = evaluate_model(trained_model, val_loader, config)

        # Record time
        train_times.append(history['total_train_time'])
        eval_times.append(eval_results['eval_time'])

        # Add time information to results
        eval_results['train_time'] = history['total_train_time']
        eval_results['eval_time'] = eval_results['eval_time']

        all_results.append(eval_results)

        # Save training history plot
        plot_training_history(
            history,
            os.path.join(config.RESULT_SAVE_DIR, f'training_history_exp_{exp_id}.png')
        )

        # Save confusion matrix
        class_names = [f'Class_{i}' for i in range(config.NUM_CLASSES)]
        plot_confusion_matrix(
            eval_results['confusion_matrix'],
            class_names,
            os.path.join(config.RESULT_SAVE_DIR, f'confusion_matrix_exp_{exp_id}.png')
        )

        # Save attention weights
        plot_attention_weights(
            eval_results['temporal_attention'],
            eval_results['feature_attention'],
            os.path.join(config.ATTENTION_SAVE_DIR, f'attention_weights_exp_{exp_id}.png')
        )

        # Record best model
        if eval_results['accuracy'] > best_accuracy:
            best_accuracy = eval_results['accuracy']
            best_exp_id = exp_id

        print(f"Experiment {exp_id + 1} results:")
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  Kappa coefficient: {eval_results['kappa']:.4f}")
        print(f"  Precision: {eval_results['precision']:.4f}")
        print(f"  Recall: {eval_results['recall']:.4f}")
        print(f"  F1 score: {eval_results['f1_score']:.4f}")
        print(f"  Training time: {history['total_train_time']:.2f} seconds")
        print(f"  Evaluation time: {eval_results['eval_time']:.2f} seconds")

    # Store time statistics
    time_stats = {
        'train_times': train_times,
        'eval_times': eval_times
    }

    # Use best model for full region classification
    print(f"\n{'=' * 70}")
    print("Starting full region classification (using best model)")
    print(f"{'=' * 70}")

    # Load full region data
    region_features, height, width, transform, crs, profile, bounds, load_time = load_region_features(config)

    print(f"\nPerforming full region classification with best model...")
    print(f"Best model from experiment {best_exp_id + 1}, accuracy: {best_accuracy:.4f}")

    # Load best model
    best_model = MultiScaleTemporalAttentionNet(
        temporal_dim=config.TEMPORAL_DIM,
        feature_dim=config.FEATURE_DIM,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    best_model.load_state_dict(
        torch.load(os.path.join(config.MODEL_SAVE_DIR, f'best_model_exp_{best_exp_id}.pth'))
    )

    # Full region prediction
    prediction_map, predict_time = predict_region(best_model, region_features, config)

    # Save prediction results
    save_time = save_prediction_map(
        prediction_map,
        transform,
        crs,
        profile,
        bounds,
        os.path.join(config.CLASSIFICATION_SAVE_DIR, 'classification_best.tif')
    )

    # Record full region classification time
    time_stats['region_classification_time'] = {
        'predict_time': predict_time,
        'save_time': save_time,
        'load_time': load_time
    }

    # Save all experimental results
    save_results_to_excel(
        all_results,
        time_stats,
        os.path.join(config.OUTPUT_DIR, 'experiment_results.xlsx')
    )

    # Save time statistics separately
    save_time_statistics_to_csv(
        time_stats,
        os.path.join(config.TIME_SAVE_DIR, 'time_statistics.csv')
    )

    # Generate final report
    overall_time = time.time() - overall_start_time
    print("\n" + "=" * 70)
    print("Experiment completed!")
    print("=" * 70)

    # Calculate average performance
    accuracies = [r['accuracy'] for r in all_results]
    kappas = [r['kappa'] for r in all_results]
    f1_scores = [r['f1_score'] for r in all_results]

    print(f"\nMSTA-Net average performance:")
    print(f"  Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  Kappa coefficient: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
    print(f"  F1 score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    print(f"\nTime efficiency analysis:")
    avg_train_time = np.mean(train_times)
    avg_eval_time = np.mean(eval_times)
    region_time = time_stats['region_classification_time']
    total_time = avg_train_time + avg_eval_time + region_time['predict_time'] + region_time['save_time'] + \
                 region_time['load_time']

    print(f"  Average training time: {avg_train_time:.2f} seconds")
    print(f"  Average evaluation time: {avg_eval_time:.2f} seconds")
    print(f"  Full region classification time: {region_time['predict_time']:.2f} seconds")
    print(f"  Total time: {total_time:.2f} seconds")

    print(f"\nTotal experiment time: {overall_time:.2f} seconds ({overall_time / 60:.2f} minutes)")
    print(f"All results saved to: {config.OUTPUT_DIR}")
    print(f"Classification results saved to: {config.CLASSIFICATION_SAVE_DIR}")
    print(f"Time statistics saved to: {config.TIME_SAVE_DIR}")
    print(f"Attention weights saved to: {config.ATTENTION_SAVE_DIR}")


if __name__ == "__main__":
    main()