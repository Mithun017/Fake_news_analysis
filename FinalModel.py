#comparative text classification using statistical and embedding-based models 

import os
import zipfile
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

np.random.seed(42)
torch.manual_seed(42)

def download_kaggle_dataset():
    """Automatically download the Fake and Real News dataset from Kaggle."""
    dataset_name = "clmentbisaillon/fake-and-real-news-dataset"
    try:
        os.system(f"kaggle datasets download -d {dataset_name} -p ./")
        
        zip_path = "./fake-and-real-news-dataset.zip"
        if not os.path.exists(zip_path):
            raise FileNotFoundError("Dataset zip file not downloaded correctly. Check Kaggle API setup.")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("./dataset")
        
        fake_df = pd.read_csv("./dataset/Fake.csv")
        true_df = pd.read_csv("./dataset/True.csv")
        
        fake_df['label'] = 0  # Fake news
        true_df['label'] = 1  # Real news
        
        df = pd.concat([fake_df, true_df], ignore_index=True)
        
        df = df[['text', 'label']].dropna()
        
        return df
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

print("Downloading and loading dataset...")
try:
    df = download_kaggle_dataset()
    print(f"Dataset shape: {df.shape}")
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit(1)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text_statistical(text):
    """Preprocess text for statistical models (bag-of-words/TF-IDF)."""
    try:
    
        text = text.lower()
        
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation and word not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

df['text_processed'] = df['text'].apply(preprocess_text_statistical)

df = df[df['text_processed'] != '']
print(f"Dataset shape after preprocessing: {df.shape}")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text_processed'], df['label'], test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

print("Training baseline Naive Bayes model...")
vectorizer_bow = CountVectorizer(max_features=5000)
X_train_bow = vectorizer_bow.fit_transform(X_train)
X_test_bow = vectorizer_bow.transform(X_test)

baseline_model = MultinomialNB()
baseline_model.fit(X_train_bow, y_train)
y_pred_baseline = baseline_model.predict(X_test_bow)

baseline_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_baseline),
    'Precision': precision_score(y_test, y_pred_baseline),
    'Recall': recall_score(y_test, y_pred_baseline),
    'F1-Score': f1_score(y_test, y_pred_baseline)
}
print("Baseline (Naive Bayes) Metrics:", baseline_metrics)

print("Training Logistic Regression model...")
vectorizer_tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

lr_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1-Score': f1_score(y_test, y_pred_lr)
}
print("Logistic Regression Metrics:", lr_metrics)

print("Training BERT model...")
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
except Exception as e:
    print(f"Error loading BERT model: {e}")
    exit(1)

train_dataset = NewsDataset(X_train, y_train, tokenizer)
test_dataset = NewsDataset(X_test, y_test, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)
print(f"Using device: {device}")

optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-5)

bert_model.train()
for epoch in range(1):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = bert_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader)}")

bert_model.eval()
y_pred_bert = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        y_pred_bert.extend(preds.cpu().numpy())

bert_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_bert),
    'Precision': precision_score(y_test, y_pred_bert),
    'Recall': recall_score(y_test, y_pred_bert),
    'F1-Score': f1_score(y_test, y_pred_bert)
}
print("BERT Metrics:", bert_metrics)

metrics_df = pd.DataFrame({
    'Baseline (Naive Bayes)': baseline_metrics,
    'Logistic Regression': lr_metrics,
    'BERT': bert_metrics
})

plt.figure(figsize=(10, 6))
metrics_df.plot(kind='bar')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(title='Models')
plt.tight_layout()
plt.savefig('model_performance_comparison.png')
plt.show()

cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_lr.png')
plt.show()

cm_bert = confusion_matrix(y_test, y_pred_bert)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix - BERT')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_bert.png')
plt.show()

metrics_df.to_csv('model_performance.csv')
print("Results and visualizations saved.")