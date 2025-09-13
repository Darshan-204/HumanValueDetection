import gradio as gr
import torch
import os
import numpy as np
import re
import pandas as pd

# Import your model classes from the existing file
from phase2_lstm_bilstm_gru import (
    LSTMClassifier,
    BiLSTMClassifier,
    GRUClassifier,
    TextCNNClassifier,
    simple_tokenize,
    LABEL_COLUMNS,
    SAVE_DIR,
    DEVICE,
    PAD_IDX,
    UNK_IDX
)

def predict_text_or_csv(model_name, text_input, csv_file):
    models_to_load = {
        'LSTM': os.path.join(SAVE_DIR, 'LSTM', 'LSTM.pt'),
        'BiLSTM': os.path.join(SAVE_DIR, 'BiLSTM', 'BiLSTM.pt'),
        'GRU': os.path.join(SAVE_DIR, 'GRU', 'GRU.pt'),
        'TextCNN': os.path.join(SAVE_DIR, 'TextCNN', 'TextCNN.pt')
    }

    if model_name not in models_to_load:
        return f"Invalid model: {model_name}"

    model_path = models_to_load[model_name]

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model_config = checkpoint['config']
    model_stoi = checkpoint['stoi']
    model_itos = checkpoint['itos']

    # Choose model
    if model_name == 'LSTM':
        model = LSTMClassifier(len(model_itos), num_labels=len(LABEL_COLUMNS))
    elif model_name == 'BiLSTM':
        model = BiLSTMClassifier(len(model_itos), num_labels=len(LABEL_COLUMNS))
    elif model_name == 'GRU':
        model = GRUClassifier(len(model_itos), num_labels=len(LABEL_COLUMNS))
    else:
        model = TextCNNClassifier(len(model_itos), num_labels=len(LABEL_COLUMNS))

    model.load_state_dict(checkpoint['model_state'])
    model = model.to(DEVICE).eval()

    # Handle text or CSV input
    if csv_file is not None:
        df = pd.read_csv(csv_file.name)
        if "clean_body" not in df.columns:
            return "CSV must contain 'clean_body' column."
        texts = df["clean_body"].astype(str).tolist()
    else:
        texts = [text_input.strip()]

    # Tokenize
    numericalized_texts = []
    for text in texts:
        toks = simple_tokenize(text)
        ids = [model_stoi.get(tok, UNK_IDX) for tok in toks][:model_config['max_len']]
        if len(ids) < model_config['max_len']:
            ids += [PAD_IDX] * (model_config['max_len'] - len(ids))
        numericalized_texts.append(np.array(ids, dtype=np.int64))

    input_tensor = torch.tensor(numericalized_texts).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    # Format result
    results = []
    for i, text in enumerate(texts):
        pred_labels = {LABEL_COLUMNS[j]: float(probs[i][j]) for j in range(len(LABEL_COLUMNS))}
        results.append(pred_labels)

    return results if csv_file else results[0]

# Gradio Interface
interface = gr.Interface(
    fn=predict_text_or_csv,
    inputs=[
        gr.Radio(["LSTM", "BiLSTM", "GRU", "TextCNN"], label="Select Model"),
        gr.Textbox(label="Enter Text Here", lines=5, placeholder="Paste email text here..."),
        gr.File(label="Upload CSV File (.csv)")
    ],
    outputs="json",
    title="Deep Email Analytics (LSTM / BiLSTM / GRU / TextCNN)"
)

if __name__ == "__main__":
    interface.launch()
