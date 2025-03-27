import pandas as pd
import torch
import torch.nn as nn
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from torch.optim import RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_optimizer import Lookahead

# Download required nltk data
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """Lowercase, remove punctuation, and remove stopwords."""
    if pd.isna(text):  # Check for NaN values
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_and_clean_data(file_path):
    """Load CSV, remove 'Other' and empty labels, clean text."""
    df = pd.read_csv(file_path)
    
    # Convert labels to lowercase and drop rows with empty labels
    df['our rating'] = df['our rating'].str.lower().str.strip()
    df = df.dropna(subset=['our rating'])  # Remove empty labels

    # Filter out unwanted labels
    valid_labels = ['true', 'false', 'partially false']
    df = df[df['our rating'].isin(valid_labels)]

    # Map labels: true=1, false=0, partially false=2
    df['our rating'] = df['our rating'].map({'true': 1, 'false': 0, 'partially false': 2})

    # Clean text columns
    df['text'] = df['text'].apply(clean_text)
    df['title'] = df['title'].apply(clean_text)

    return df

def balance_data(df):
    df_0 = df[df['our rating'] == 0]
    df_1 = df[df['our rating'] == 1]
    df_2 = df[df['our rating'] == 2]
    max_len = max(len(df_0), len(df_1), len(df_2))
    df_0_resampled = resample(df_0, replace=True, n_samples=max_len, random_state=42)
    df_1_resampled = resample(df_1, replace=True, n_samples=max_len, random_state=42)
    df_2_resampled = resample(df_2, replace=True, n_samples=max_len, random_state=42)
    return pd.concat([df_0_resampled, df_1_resampled, df_2_resampled])

class FuzzyLogicLayer(nn.Module):
    def __init__(self):
        super(FuzzyLogicLayer, self).__init__()
        self.weights = nn.Parameter(torch.tensor([10.0, 10.0, 10.0], requires_grad=True, device="cuda"))
    
    def forward(self, logits):
        softmax = nn.Softmax(dim=1)
        probs = softmax(logits)
        true_mf = torch.tensor([0.7, 0.85, 1.0], device=logits.device)
        false_mf = torch.tensor([0.0, 0.15, 0.3], device=logits.device)
        partial_mf = torch.tensor([0.25, 0.5, 0.75], device=logits.device)
        true_score = (probs[:, 2] - true_mf[0]) / (true_mf[1] - true_mf[0])
        false_score = (probs[:, 0] - false_mf[0]) / (false_mf[1] - false_mf[0])
        partial_score = (probs[:, 1] - partial_mf[0]) / (partial_mf[1] - partial_mf[0])
        true_score *= self.weights[0]
        partial_score *= self.weights[1]
        false_score *= self.weights[2]
        fuzzy_scores = torch.stack([false_score, partial_score, true_score], dim=1)
        return fuzzy_scores

class MultiFakeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=4096):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        title = self.data.iloc[idx]['title']
        label = self.data.iloc[idx]['our rating']
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

def custom_collate_fn(batch):
    input_ids, attention_masks, labels = zip(*batch)
    return torch.stack(input_ids), torch.stack(attention_masks), torch.stack(labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'allenai/longformer-base-4096'
tokenizer = LongformerTokenizer.from_pretrained(MODEL_NAME)
model = LongformerForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
fuzzy_layer = FuzzyLogicLayer().to(device)

train_df = load_and_clean_data('multifaketrain.csv')
test_df = load_and_clean_data('multifaketest.csv')
train_df = balance_data(train_df)

train_dataset = MultiFakeDataset(train_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)
test_dataset = MultiFakeDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)


num_epochs = 3
fuzzy_layer.weights.requires_grad = True

for k in k_values:
    for alpha in alpha_values:
        print(f"Training with k={k}, alpha={alpha}")

        model = LongformerForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
        base_optimizer = optim.AdamW(model.parameters(), lr=1e-5,weight_decay=0.01)
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.7)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
        scaler = GradScaler()

        num_epochs = 3
        for epoch in range(num_epochs):
            model.train()
            total_loss, all_preds, all_labels = 0, [], []

            for input_ids, attention_mask, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)

                scaler.scale(loss).backward()
                scaler.step(base_optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_loss += loss.item()
                all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Compute training metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted')
            f1 = f1_score(all_labels, all_preds, average='weighted')

            print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, '
                  f'Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}')

        # Test loop
        model.eval()
        all_preds, all_labels, total_loss = [], [], 0

        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(test_loader, desc='Testing'):
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()
                all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute test metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f'Test Results - k={k}, alpha={alpha}')
        print(f'Loss: {total_loss/len(test_loader):.4f}, '
              f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

