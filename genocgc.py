# MIT License
# 
# Copyright (c) 2025 Anas Abu-Doleh
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.gt import GraphormerLayer
from sklearn.metrics import matthews_corrcoef
import numpy as np
import itertools
SEGMENT_LENGTH = 32
NUM_SEGMENTS = 0
EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_CLASSES = 2
NUM_EPOCHS = 150
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
TRANS_SPLIT_COUNT = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------
# Function to parse .fna genomic sequence files and extract segments
# ----------------------
def parse_fna(file_path, segment_length=SEGMENT_LENGTH,  include_reverse_complement=False):
    global NUM_SEGMENTS,TRANS_SPLIT_COUNT
    sequences, labels = [], []
    with open(file_path, "r") as f:
        current_seq, current_label = "", None
        for line in f:
            if line.startswith(">"):
                if current_seq:
                    NUM_SEGMENTS = 2*(len(current_seq) //segment_length)
                    step = max(1, (len(current_seq) - segment_length) // (NUM_SEGMENTS - 1))
                    segments = [current_seq[i:i+segment_length] for i in range(0, step * NUM_SEGMENTS, step)][:NUM_SEGMENTS]
                    sequences.append(segments)
                    labels.append(current_label)
                current_label = int(line.strip().split("|")[-1]) if "|" in line else 0
                current_seq = ""
            else:
                current_seq += line.strip()
        if current_seq:
            NUM_SEGMENTS = 2*(len(current_seq) //segment_length)
            step = max(1, (len(current_seq) - segment_length) // (NUM_SEGMENTS- 1))
            segments = [current_seq[i:i+segment_length] for i in range(0, step * NUM_SEGMENTS, step)][:NUM_SEGMENTS]
            sequences.append(segments)
            labels.append(current_label)

    if  include_reverse_complement:
        complement = str.maketrans("ACGT", "TGCA")
        rev_sequences = [[segment.translate(complement)[::-1] for segment in seq] for seq in sequences]
        sequences.extend(rev_sequences)
        labels.extend(labels)  # Duplicate labels for reverse complements
    TRANS_SPLIT_COUNT = NUM_SEGMENTS//5
    return sequences, labels
# ----------------------
# Local CNN module to extract local features from each DNA segment
# ----------------------
class LocalCNN(nn.Module):
    def __init__(self, output_dim):
        super(LocalCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64*2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64*2, output_dim, kernel_size=3,stride=1, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return x
NUM_GRAPHORMER_LAYERS=2
KMER_DIM=0
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

# ----------------------
# Main DNAClassifier: Combines Local CNN + Graphormer + FC layers
# ----------------------
class DNAClassifier(nn.Module):
    def to_device(self):
        self.to(device)
        for module in self.modules():
            if isinstance(module, nn.Module):
                module.to(device)

    def __init__(self, in_channels=4, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, num_classes=NUM_CLASSES):
        super(DNAClassifier, self).__init__()
        self.local_cnns = nn.ModuleList([LocalCNN(embedding_dim) for _ in range(NUM_SEGMENTS)])
        self.graphormer_layers = nn.ModuleList([
            GraphormerLayer(embedding_dim, hidden_size=2 * embedding_dim, num_heads=num_heads)
            for _ in range(NUM_GRAPHORMER_LAYERS)
        ])
        self.fc = nn.Linear(embedding_dim * TRANS_SPLIT_COUNT, num_classes)

    def compute_kmer_frequencies(self, x, k=2):
        """Compute K-mer frequency distribution for each segment in x."""
        batch_size, num_segments, _, segment_length = x.shape
        kmer_dict = {"".join(kmer): i for i, kmer in enumerate(itertools.product("ACGT", repeat=k))}
        num_kmers = len(kmer_dict)
        nucleotide_indices = x.argmax(dim=2)  # (B, S, L)
        kmer_counts = torch.zeros((batch_size, num_segments, num_kmers), device=x.device)
        for i in range(segment_length - k + 1):
            kmer_indices = sum([nucleotide_indices[:, :, i+j] * (4 ** (k - j - 1)) for j in range(k)])  # Unique k-mer keys
            for km in range(num_kmers):
                kmer_counts[:, :, km] += (kmer_indices == km).float()
        kmer_counts /= kmer_counts.sum(dim=-1, keepdim=True) + 1e-6  # Avoid division by zero
        return kmer_counts  # (B, S, num_kmers)

    def compute_bias_matrices(self, combined_features):
        batch_size, num_segments, _ = combined_features.shape
        distance_matrix = torch.cdist(combined_features, combined_features, p=2)
        distance_matrix = distance_matrix / (distance_matrix.max() + 1e-6)  # Normalize to (0,1)
        indices = torch.arange(num_segments, device=combined_features.device).float()
        real_distance_matrix = torch.abs(indices[:, None] - indices[None, :]) / (num_segments - 1)  # (S, S)
        real_distance_matrix = real_distance_matrix.unsqueeze(0).expand(batch_size, -1, -1)  # Expand to (B, S, S)
        inverse_distance_matrix = (1 - real_distance_matrix)  # Already expanded
        inverse_real_distance_matrix = (1 - distance_matrix)  # Already expanded
        bias_matrices = torch.stack([distance_matrix, real_distance_matrix,inverse_distance_matrix,inverse_real_distance_matrix], dim=-1)

        return bias_matrices  # (B, S, S, 2)

    def forward(self, x):
        batch_size, num_segments, in_channels, seq_length = x.shape
        kmer_frequencies = self.compute_kmer_frequencies(x, k=2)
        cnn_features = torch.stack([self.local_cnns[i](x[:, i, :, :]) for i in range(NUM_SEGMENTS)], dim=1)
        cnn_features = F.layer_norm(cnn_features, cnn_features.shape[-1:])  # Normalize

        combined_features = cnn_features  # Only segmentation embeddings
        attn_bias = self.compute_bias_matrices(combined_features)  # (B, S, S, 4)
        for layer in self.graphormer_layers:
            combined_features = layer(combined_features, attn_bias)
        num_splits = TRANS_SPLIT_COUNT
        sequence_length = combined_features.shape[1]
        split_size = sequence_length // num_splits
        remaining = sequence_length % num_splits
        segments = []
        start_idx = 0
        for i in range(num_splits):
            end_idx = start_idx + split_size
            if i == num_splits - 1:
                end_idx += remaining
            segment = combined_features[:, start_idx:end_idx, :]
            pooled = torch.max(segment, dim=1)[0]
            segments.append(pooled)
            start_idx = end_idx
        final_representation = torch.cat(segments, dim=1)

        x = self.fc(final_representation)
        return x
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
# Compute Matthews Correlation Coefficient for evaluation
def compute_mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)

from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import torch

# Compute Matthews Correlation Coefficient for evaluation
def compute_mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)

# ----------------------
# Training loop with early stopping based on MCC
# ----------------------
def train_model(model, train_loader, val_loader, test_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, patience=40):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_mcc = -1.0  # Track the best validation MCC
    epochs_without_improvement = 0  # Track epochs without MCC improvement

    for epoch in range(num_epochs):
        model.train()
        train_preds, train_labels = [], []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        train_mcc = compute_mcc(train_labels, train_preds)
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        val_mcc = compute_mcc(val_labels, val_preds)
        test_preds, test_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                test_labels.extend(batch_y.cpu().numpy())

        test_mcc = compute_mcc(test_labels, test_preds)

        print(f"Epoch {epoch+1}/{num_epochs} - Train MCC: {train_mcc:.4f}, Val MCC: {val_mcc:.4f}, Test MCC: {test_mcc:.4f}")
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            epochs_without_improvement = 0  # Reset counter if there's an improvement
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Stopping early at epoch {epoch+1} due to no improvement in MCC for {patience} epochs.")
            break
import random
import numpy as np
import torch
# ----------------------
# Seed setter for reproducibility
# ----------------------
def set_random_seed(seed=42):
    """
    Set seed for reproducibility in Python, NumPy, and PyTorch.
    """
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (if available)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Avoids non-deterministic optimizations

# ----------------------
# Script entry point
# ----------------------
if __name__ == "__main__":
    set_random_seed(59)
    train_sequences, train_labels = parse_fna("train.fna",include_reverse_complement=False)
    test_sequences, test_labels = parse_fna("test.fna")
    def encode_one_value(sequences):
        mapping = {'A': 0.25, 'C': 0.5, 'G': 0.75, 'T': 1}
        encoded = []
        for seq_set in sequences:
            segment_tensors = [torch.tensor([mapping.get(nuc, 0.0) for nuc in segment], dtype=torch.float32).unsqueeze(0) for segment in seq_set]
            encoded.append(torch.stack(segment_tensors))
        return torch.stack(encoded)

    def encode_one_hot(sequences):
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
        encoded = []
        for seq_set in sequences:
            segment_tensors = [torch.tensor([mapping.get(nuc, [0, 0, 0, 0]) for nuc in segment], dtype=torch.float32).T for segment in seq_set]
            encoded.append(torch.stack(segment_tensors))
        return torch.stack(encoded)
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_sequences, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )
    train_data = encode_one_hot(train_data)
    val_data = encode_one_hot(val_data)
    test_data = encode_one_hot(test_sequences)
    train_loader = [(train_data[i:i+BATCH_SIZE], torch.tensor(train_labels[i:i+BATCH_SIZE], dtype=torch.long)) for i in range(0, len(train_data), BATCH_SIZE)]
    val_loader = [(val_data[i:i+BATCH_SIZE], torch.tensor(val_labels[i:i+BATCH_SIZE], dtype=torch.long)) for i in range(0, len(val_data), BATCH_SIZE)]
    test_loader = [(test_data[i:i+BATCH_SIZE], torch.tensor(test_labels[i:i+BATCH_SIZE], dtype=torch.long)) for i in range(0, len(test_data), BATCH_SIZE)]
    model = DNAClassifier().to(device)
    model.to_device()
    train_model(model, train_loader, val_loader, test_loader)
