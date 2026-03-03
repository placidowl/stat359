import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


# =========================
# Config
# =========================
DATA_PATH = "Bert_data/combined_pirate_dataset.json"
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "pirate_bert_output"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
RANDOM_SEED = 816


# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)


# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =========================
# Load JSON data
# Expected format:
# [
#   {"sentence": "...", "label": 1},
#   {"sentence": "...", "label": 0}
# ]
# =========================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

for item in data:
    text = item.get("sentence", "").strip()
    label = item.get("label", None)

    if text == "" or label is None:
        continue

    texts.append(text)
    labels.append(int(label))

print(f"Loaded {len(texts)} examples.")


# =========================
# Split data
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.30, stratify=labels, random_state=RANDOM_SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
)

print(f"Train size: {len(X_train)}")
print(f"Val size: {len(X_val)}")
print(f"Test size: {len(X_test)}")


# =========================
# Tokenizer
# =========================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


# =========================
# Dataset
# =========================
class PirateDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


train_dataset = PirateDataset(X_train, y_train, tokenizer, MAX_LEN)
val_dataset = PirateDataset(X_val, y_val, tokenizer, MAX_LEN)
test_dataset = PirateDataset(X_test, y_test, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# Model
# =========================
class BertBinaryClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits


model = BertBinaryClassifier(MODEL_NAME).to(device)


# =========================
# Loss / optimizer
# =========================
class_counts = np.bincount(y_train)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=LR)


# =========================
# Train / eval helpers
# =========================
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")

    return avg_loss, acc, f1, all_preds, all_labels


# =========================
# Training loop
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

best_val_f1 = -1.0
best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")

for epoch in range(EPOCHS):
    train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, val_f1, _, _ = run_epoch(model, val_loader, criterion)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Saved new best model to {best_model_path}")


# =========================
# Final test evaluation
# =========================
print("\nLoading best model for final test evaluation...")
model.load_state_dict(torch.load(best_model_path, map_location=device))

test_loss, test_acc, test_f1, test_preds, test_labels = run_epoch(model, test_loader, criterion)

print("\n===== TEST RESULTS =====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Acc : {test_acc:.4f}")
print(f"Test F1  : {test_f1:.4f}")
print(classification_report(test_labels, test_preds, digits=4))

print(f"\nDone. Model + tokenizer saved in: {OUTPUT_DIR}")