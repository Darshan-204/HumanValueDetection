title: Human Value
emoji: 💻
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.45.0
app_file: app.py
pinned: false

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Human Value Detection

This project provides a Gradio web interface for detecting human values in text using deep learning models (LSTM, BiLSTM, GRU, TextCNN). It is deployed as a Hugging Face Space: [Human_Value](https://huggingface.co/spaces/darshan204/Human_Value).

## Features
- Predict human values from text or CSV files
- Supports multiple models: LSTM, BiLSTM, GRU, TextCNN
- Easy-to-use Gradio interface

## Usage
### Online
Visit the Hugging Face Space: [Human_Value](https://huggingface.co/spaces/darshan204/Human_Value)

### Local
Clone the repository and install requirements:
```bash
git clone https://github.com/Darshan-204/HumanValueDetection.git
cd HumanValueDetection
pip install -r requirements.txt
python app.py
```

## Model Artifacts
Pretrained models are stored in the `artifacts/` directory:
- `artifacts/LSTM/LSTM.pt`
- `artifacts/BiLSTM/BiLSTM.pt`
- `artifacts/GRU/GRU.pt`
- `artifacts/TextCNN/TextCNN.pt`

## License
This project is for research and educational purposes.
