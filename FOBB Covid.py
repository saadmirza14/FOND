import pandas as pd
import torch
import torch.nn as nn
import re
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch.optim as optim
from sklearn.utils import resample
from torch_optimizer import Lookahead

# Load the data
df = pd.read_excel('COVID_fake_new_dataset.xlsx')

# Map labels to numerical values
df['label'] = df['subcategory'].apply(
    lambda x: 0 if x == 'false news' else (1 if x == 'partially false' else 2)
)

# Clean the text data
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply text cleaning
df['text'] = df['text'].apply(clean_text)
df['title'] = df['title'].apply(clean_text)

# Balance the dataset
true_news = df[df['label'] == 2]
false_news = df[df['label'] == 0]
partially_false = df[df['label'] == 1]

majority_size = len(true_news)
false_news_upsampled = resample(false_news, replace=True, n_samples=majority_size, random_state=42)
partially_false_upsampled = resample(partially_false, replace=True, n_samples=majority_size, random_state=42)

balanced_df = pd.concat([true_news, false_news_upsampled, partially_false_upsampled])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train and validation sets
train_data, val_data = train_test_split(balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['label'])

# Define dataset class
class COVID19FakeNewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=4096):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        title = self.data.iloc[idx]['title']
        combined_text = text + " " + title
        inputs = self.tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

# Define Fuzzy Logic Layer
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

# Initialize model and tokenizer
MODEL_NAME = 'google/bigbird-roberta-base'
tokenizer = BigBirdTokenizer.from_pretrained(MODEL_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BigBirdForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
fuzzy_layer = FuzzyLogicLayer().to(device)

# Create DataLoaders
train_loader = DataLoader(COVID19FakeNewsDataset(train_data, tokenizer), batch_size=2, shuffle=True, pin_memory=True)
val_loader = DataLoader(COVID19FakeNewsDataset(val_data, tokenizer), batch_size=2, shuffle=False, pin_memory=True)

# Training setup
base_optimizer = optim.AdamW([
    {'params': model.parameters(), 'lr': 1e-5},
    {'params': fuzzy_layer.parameters(), 'lr': 1e-5, 'weight_decay': 0.01}
])
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler() 
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for input_ids, attention_mask, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            fuzzy_outputs = fuzzy_layer(outputs)
            loss = criterion(fuzzy_outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
        # Collect predictions and labels
        preds = torch.argmax(fuzzy_outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute training metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, '
          f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

total_loss = 0
all_preds, all_labels = [], []
model.eval()
fuzzy_layer.eval()

with torch.no_grad():
    for input_ids, attention_mask, labels in tqdm(val_loader, desc='Testing'):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        fuzzy_preds = fuzzy_layer(outputs)
        
        # Collect predictions and labels
        all_preds.extend(torch.argmax(fuzzy_preds, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute testing metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Testing Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
