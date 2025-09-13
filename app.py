# app.py
import os
import re
import torch
import numpy as np
import pandas as pd
import gradio as gr
from torch import nn
import torch.nn.functional as F

# =====================
# Config
# =====================
TEXT_COL = "clean_body"
LABEL_COLUMNS = [
    "power", "achievement", "hedonism", "stimulation", "self_direction",
    "security", "conformity", "tradition", "benevolence", "universalism"
]
SAVE_DIR = "artifacts"
MAX_LEN = 256
PAD_IDX = 0
UNK_IDX = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Model Definitions
# =====================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden=128, layers=2, dropout=0.3, num_labels=10):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_labels)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.drop(out)
        return self.fc(out)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden=128, layers=2, dropout=0.4, num_labels=10):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_labels)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.drop(out)
        return self.fc(out)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden=128, layers=2, dropout=0.3, num_labels=10):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_labels)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.drop(out)
        return self.fc(out)


class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, filter_sizes=[3, 4, 5], num_filters=100, dropout=0.5, num_labels=10):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_labels)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.emb(x)
        x = x.permute(0, 2, 1)
        x = [self.conv_and_pool(x, conv) for conv in self.convs]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# =====================
# Utils
# =====================
def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

def load_model(model_name):
    models_to_load = {
        "LSTM": os.path.join(SAVE_DIR, "LSTM", "LSTM.pt"),
        "BiLSTM": os.path.join(SAVE_DIR, "BiLSTM", "BiLSTM.pt"),
        "GRU": os.path.join(SAVE_DIR, "GRU", "GRU.pt"),
        "TextCNN": os.path.join(SAVE_DIR, "TextCNN", "TextCNN.pt")
    }

    if model_name not in models_to_load:
        raise ValueError(f"Invalid model: {model_name}")

    checkpoint = torch.load(models_to_load[model_name], map_location=DEVICE)
    model_config = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    if model_name == "LSTM":
        model = LSTMClassifier(len(itos), num_labels=len(LABEL_COLUMNS))
    elif model_name == "BiLSTM":
        model = BiLSTMClassifier(len(itos), num_labels=len(LABEL_COLUMNS))
    elif model_name == "GRU":
        model = GRUClassifier(len(itos), num_labels=len(LABEL_COLUMNS))
    else:
        model = TextCNNClassifier(len(itos), num_labels=len(LABEL_COLUMNS))

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE).eval()
    return model, model_config, stoi

def predict_text_or_csv(model_name, text_input, csv_file):
    model, model_config, stoi = load_model(model_name)

    if csv_file is not None:
        df = pd.read_csv(csv_file.name)
        if TEXT_COL not in df.columns:
            return f"CSV must have '{TEXT_COL}' column."
        texts = df[TEXT_COL].astype(str).tolist()
    else:
        texts = [text_input.strip()]

    numericalized_texts = []
    for text in texts:
        toks = simple_tokenize(text)
        ids = [stoi.get(tok, UNK_IDX) for tok in toks][:model_config["max_len"]]
        if len(ids) < model_config["max_len"]:
            ids += [PAD_IDX] * (model_config["max_len"] - len(ids))
        numericalized_texts.append(np.array(ids, dtype=np.int64))

    input_tensor = torch.tensor(numericalized_texts).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    if csv_file:
        # ✅ Return average predictions across all rows
        avg_probs = probs.mean(axis=0)
        avg_labels = {LABEL_COLUMNS[j]: float(avg_probs[j]) for j in range(len(LABEL_COLUMNS))}
        return avg_labels
    else:
        pred_labels = {LABEL_COLUMNS[j]: float(probs[0][j]) for j in range(len(LABEL_COLUMNS))}
        return pred_labels

# =====================
# Gradio App
# =====================
interface = gr.Interface(
    fn=predict_text_or_csv,
    inputs=[
        gr.Radio(["LSTM", "BiLSTM", "GRU", "TextCNN"], label="Select Model"),
        gr.Textbox(label="Enter Text", lines=5, placeholder="Paste email text..."),
        gr.File(label="Upload CSV File (.csv)")
    ],
    outputs="json",
    title="Deep Email Analytics - Phase 2 Models"
)

if __name__ == "__main__":
    interface.launch()
