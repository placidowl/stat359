import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive pair

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        center, context = self.pairs[index]
        return torch.tensor(center, dtype = torch.long), \
               torch.tensor(context, dtype = torch.long)


# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, EMBEDDING_DIM):
        super().__init__()
        self.center_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_DIM)
        self.context_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_DIM)

    def forward(self, center_word, context_word):
        center_embedding = self.center_embed(center_word)
        context_embedding = self.context_embed(context_word)
        return torch.sum(center_embedding * context_embedding, dim = 1)
    
    def get_embeddings(self):
        return self.center_embed.weight.detach().cpu().numpy()
    
        

# Load processed data
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Precompute negative sampling distribution below
vocab_size = len(data['word2idx'])

counts = torch.zeros(vocab_size)
for word, count in data['counter'].items():
    idx = data['word2idx'][word]
    counts[idx] = float(count)

neg_sampling_dist = counts ** 0.75
neg_sampling_dist = neg_sampling_dist / neg_sampling_dist.sum()

neg_samples = torch.multinomial(neg_sampling_dist, num_samples=NEGATIVE_SAMPLES * BATCH_SIZE, replacement=True)


# Device selection: CUDA > MPS > CPU

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")



# Dataset and DataLoader
pairs = list(zip(data['skipgram_df']['center'].tolist(),
                 data['skipgram_df']['context'].tolist()))
dataset = SkipGramDataset(pairs)

val_frac = 0.05
val_size = int(len(dataset) * val_frac)
train_size = len(dataset) - val_size

from torch.utils.data import random_split

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size= BATCH_SIZE,
    shuffle= True
)

val_loader = DataLoader(
    val_dataset,
    batch_size= BATCH_SIZE,
    shuffle= False
)

# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

def make_targets(center, context, vocab_size):
    batch_size = center.size(0)
    pos_labels = torch.ones(batch_size, device = center.device)
    neg_labels = torch.zeros(batch_size * NEGATIVE_SAMPLES, device = center.device)
    return pos_labels, neg_labels

def sgns_batch_loss(center, context):
    center = center.to(device)
    context = context.to(device)

    batch_size = center.size(0)

    pos_labels, neg_labels = make_targets(center, context, vocab_size)

    pos_logits = model(center, context)
    pos_loss = criterion(pos_logits, pos_labels)

    neg_context = torch.multinomial(
        neg_sampling_dist,
        num_samples=batch_size * NEGATIVE_SAMPLES,
        replacement=True
    ).view(batch_size, NEGATIVE_SAMPLES).to(device)

    center_rep = center.unsqueeze(1).expand_as(neg_context)

    neg_logits = model(center_rep.reshape(-1), neg_context.reshape(-1))
    neg_logits = neg_logits.view(-1)
    neg_labels = neg_labels.view(-1)

    neg_loss = criterion(neg_logits, neg_labels)

    return pos_loss + neg_loss


# Training loop
import matplotlib.pyplot as plt

num_epochs = EPOCHS
train_losses = []
val_losses = []

model.train()  # Set model to training mode

for epoch in range(num_epochs):
    epoch_train_loss = 0.0
    num_train_batches = 0
    
    # Training loop
    for center, context in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
        
        loss = sgns_batch_loss(center,context)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        num_train_batches += 1
    avg_train_loss = epoch_train_loss / max(1, num_train_batches)
    train_losses.append(avg_train_loss)

    # Validation loop
    model.eval()
    epoch_val_loss = 0.0
    num_val_batches = 0
    with torch.no_grad():
        for center, context in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]"):
            
            val_loss = sgns_batch_loss(center, context)
            epoch_val_loss += val_loss.item()
            num_val_batches += 1
    avg_val_loss = epoch_val_loss / max(1, num_val_batches)
    val_losses.append(avg_val_loss)
    model.train()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")


