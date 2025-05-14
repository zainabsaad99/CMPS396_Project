import streamlit as st
import torch
import torch.nn as nn
import math
import pandas as pd
import string
import numpy as np
import gdown
import os
from tqdm import tqdm

# Custom theme colors based on the slide
BACKGROUND_COLOR = "#FFF8EC"  # Cream/beige background
PRIMARY_COLOR = "#7E57C2"     # Main purple color
SECONDARY_COLOR = "#5E35B1"   # Darker purple for accents
TEXT_COLOR = "#412e3f"        # Updated text color

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Configuration ---
MODEL_CONFIGS = {
    "Caesar Cipher (Vanilla)": {
        "d_model": 512,
        "num_heads": 4,
        "num_layers": 6,
        "d_ff": 512,
        "dropout": 0.11034371565549608,
        "drive_id": "1bE1-002_YAsA-PYi_SZnUa0DiieLJvEW",
        "type": "vanilla",
        "max_length": 200
    },
    "Monoalphabetic (1 Key)": {
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "d_ff": 1024,
        "dropout": 0.12866753791718177,
        "drive_id": "1yBWYM4g4yZeFsvoKKSk3bKgU5OdYE01t",
        "type": "vanilla",
        "max_length": 256
    },
    "Monoalphabetic (5 Keys)": {
        "d_model": 512,
        "num_heads": 2,
        "num_layers": 6,
        "d_ff": 256,
        "dropout": 0.21983729110648678,
        "drive_id": "1FS-ztvYOvcTJZAdze8zV7AwArj7JiaLr",
        "type": "vanilla",
        "max_length": 256
    },
    "Monoalphabetic (100 Keys)": {
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 8,
        "d_ff": 256,
        "dropout": 0.13711053703933423,
        "drive_id": "1jF1tk0BAgoishJKb6M4X8-UcRcwJURgI",
        "type": "vanilla",
        "max_length": 256
    },
    "Vigenère (1 Key)": {
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "d_ff": 256,
        "dropout": 0.1125331367392462,
        "drive_id": "1eaOssXblkkTxyRLZLYql1cgq-fpJnOwR",
        "type": "vanilla",
        "max_length": 256
    },
    "Vigenère (5 Keys)": {
        "d_model": 256,
        "num_heads": 2,
        "num_layers": 6,
        "d_ff": 512,
        "dropout": 0.10213470498440091,
        "drive_id": "1zZANocgPpGc-m-cGM29c37f93NLKmcJD",
        "type": "vanilla",
        "max_length": 256
    },
    "Vigenère (100 Keys)": {
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 8,
        "d_ff": 1024,
        "dropout": 0.1018942343827998,
        "drive_id": "1fZaSKi8CbR74Ft7FszXL-T8nVzkkgkSY",
        "type": "vanilla",
        "max_length": 256
    },
    "Rail Fence Cipher": {
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 2,
        "d_ff": 512,
        "dropout": 0.1861,
        "drive_id": "1sIgjFeWySFV9JnW_OfhlZ3uilJkGSgN7",
        "type": "rail_fence",
        "max_length": 512
    },
    "Caesar Cipher (Enhanced)": {
        "d_model": 256,
        "num_heads": 4,
        "num_layers": 4,
        "d_ff": 256,
        "dropout": 0.2,
        "drive_id": "196CqtcO3gqO7lDzy1IKGvkLnPOyFjXPl",
        "type": "enhanced_caesar",
        "max_length": 256
    }
}

# --- Vocabulary Class ---
class Vocabulary:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.unk_token = 3
        self._build_vocab()

    def _build_vocab(self):
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        all_chars = list(string.printable)
        self.char2idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.char2idx.update({char: idx+len(special_tokens) for idx, char in enumerate(all_chars)})
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def __len__(self):
        return len(self.char2idx)

    def encode(self, text):
        if isinstance(text, float):  # Handle NaN values
            text = str(text)
        return [self.char2idx.get(char, self.unk_token) for char in text]

    def decode(self, indices):
        return ''.join([self.idx2char.get(idx, '<UNK>') for idx in indices if idx not in {self.pad_token, self.sos_token, self.eos_token}])

# --- Transformer Components ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_probs = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_probs, V)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.combine_heads(attn_output))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_output))

# --- Vanilla Transformer Model ---
class VanillaTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        return self.fc(dec_output)

# --- Rail Fence Transformer Model ---
class RailFenceTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        
        # Change these to match the trained model
        self.enc_layers = nn.ModuleList([
            self._build_encoder_layer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dec_layers = nn.ModuleList([
            self._build_decoder_layer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def _build_encoder_layer(self, d_model, num_heads, d_ff, dropout):
        layer = nn.ModuleDict({
            'attn': MultiHeadAttention(d_model, num_heads),
            'ff': PositionWiseFeedForward(d_model, d_ff),
            'norm1': nn.LayerNorm(d_model),
            'norm2': nn.LayerNorm(d_model),
            'dropout': nn.Dropout(dropout)
        })
        return layer
    
    def _build_decoder_layer(self, d_model, num_heads, d_ff, dropout):
        layer = nn.ModuleDict({
            'self_attn': MultiHeadAttention(d_model, num_heads),
            'cross_attn': MultiHeadAttention(d_model, num_heads),
            'ff': PositionWiseFeedForward(d_model, d_ff),
            'norm1': nn.LayerNorm(d_model),
            'norm2': nn.LayerNorm(d_model),
            'norm3': nn.LayerNorm(d_model),
            'dropout': nn.Dropout(dropout)
        })
        return layer
    
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src = self.dropout(self.pos_enc(self.src_emb(src)))
        tgt = self.dropout(self.pos_enc(self.tgt_emb(tgt)))
        
        for layer in self.enc_layers:
            attn_output = layer['attn'](src, src, src, src_mask)
            src = layer['norm1'](src + layer['dropout'](attn_output))
            ff_output = layer['ff'](src)
            src = layer['norm2'](src + layer['dropout'](ff_output))
        
        for layer in self.dec_layers:
            attn_output = layer['self_attn'](tgt, tgt, tgt, tgt_mask)
            tgt = layer['norm1'](tgt + layer['dropout'](attn_output))
            attn_output = layer['cross_attn'](tgt, src, src, src_mask)
            tgt = layer['norm2'](tgt + layer['dropout'](attn_output))
            ff_output = layer['ff'](tgt)
            tgt = layer['norm3'](tgt + layer['dropout'](ff_output))
        
        return self.fc(tgt)

# --- Enhanced Caesar Transformer Model ---
class CaesarTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=4, num_layers=4, d_ff=256, max_seq_length=256, dropout=0.2):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.encoder_pos = nn.Embedding(max_seq_length, d_model)
        self.decoder_pos = nn.Embedding(max_seq_length, d_model)

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_length = max_seq_length

    def forward(self, src, tgt):
        src_mask = (src == 0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)

        src_pos = torch.arange(0, src.size(1), device=device).unsqueeze(0)
        tgt_pos = torch.arange(0, tgt.size(1), device=device).unsqueeze(0)

        src_embedded = self.dropout(self.encoder_embedding(src) + self.encoder_pos(src_pos))
        tgt_embedded = self.dropout(self.decoder_embedding(tgt) + self.decoder_pos(tgt_pos))

        memory = src_embedded
        for layer in self.encoder_layers:
            memory = layer(memory, src_key_padding_mask=src_mask)

        output = tgt_embedded
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)

        return self.fc(output)

# --- Model Loading Functions ---
def download_model(drive_id, model_name):
    """Download model from Google Drive if not already exists"""
    model_path = f"models/{model_name}.pth"
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(model_path):
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, model_path, quiet=False)
    return model_path

def load_vanilla_model(config, model_path):
    """Load vanilla transformer model"""
    vocab = Vocabulary()
    
    if config["type"] == "rail_fence":
        model = RailFenceTransformer(
            src_vocab_size=len(vocab),
            tgt_vocab_size=len(vocab),
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            d_ff=config["d_ff"],
            max_len=config["max_length"],
            dropout=config["dropout"]
        ).to(device)
    else:
        model = VanillaTransformer(
            src_vocab_size=len(vocab),
            tgt_vocab_size=len(vocab),
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            d_ff=config["d_ff"],
            max_seq_length=config["max_length"],
            dropout=config["dropout"]
        ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, vocab

def load_enhanced_caesar_model(config, model_path):
    """Load enhanced Caesar transformer model"""
    vocab = Vocabulary()
    model = CaesarTransformer(
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_length=config["max_length"],
        dropout=config["dropout"]
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, vocab

# --- Decryption Functions ---
def vanilla_decrypt_text(model, text, vocab, max_length, device):
    """Decryption function for vanilla transformer models"""
    model.eval()
    with torch.no_grad():
        encoded = [vocab.sos_token] + vocab.encode(text) + [vocab.eos_token]
        encoded = encoded + [vocab.pad_token] * (max_length - len(encoded))
        encoded = torch.tensor(encoded[:max_length]).unsqueeze(0).to(device)
        target = torch.tensor([[vocab.sos_token]]).to(device)

        for _ in range(max_length - 1):
            output = model(encoded, target)
            # Handle different model output formats
            if isinstance(model, RailFenceTransformer):
                next_token = output.argmax(-1)[:, -1].item()
            else:
                next_token = output.argmax(2)[:, -1].item()
                
            if next_token == vocab.eos_token:
                break
            target = torch.cat([target, torch.tensor([[next_token]]).to(device)], dim=1)

        return vocab.decode(target[0].cpu().numpy())

def enhanced_caesar_decrypt_text(model, text, vocab, max_length, device):
    """Decryption function for enhanced Caesar transformer model"""
    model.eval()
    with torch.no_grad():
        encoded = [vocab.sos_token] + vocab.encode(text) + [vocab.eos_token]
        encoded = encoded + [vocab.pad_token] * (max_length - len(encoded))
        encoded = torch.tensor(encoded[:max_length]).unsqueeze(0).to(device)

        target = torch.tensor([[vocab.sos_token]]).to(device)

        for _ in range(max_length - 1):
            output = model(encoded, target)
            next_token = output.argmax(2)[:, -1].item()
            if next_token == vocab.eos_token:
                break
            target = torch.cat([target, torch.tensor([[next_token]]).to(device)], dim=1)

        decrypted = vocab.decode(target[0].cpu().numpy())
        return decrypted

# --- Custom Streamlit Theme and Styling ---
def set_custom_theme():
    # Set page config
    st.set_page_config(
        page_title="LLMs as Cryptanalysts", 
        page_icon="🔐", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown(f"""
    <style>
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        
        .stApp {{
            background-color: {BACKGROUND_COLOR};
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: {BACKGROUND_COLOR};
            font-family: 'Georgia', serif;
        }}
        
        p, div, span, label {{
            color: {TEXT_COLOR};
        }}
        
        .stSelectbox label, .stTextArea label, .stTextInput label {{
            color: {TEXT_COLOR};
        }}
        
        .stButton>button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border-radius: 4px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }}
        
        .stButton>button:hover {{
            background-color: {SECONDARY_COLOR};
        }}
        
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
            border-color: {PRIMARY_COLOR};
            border-radius: 4px;
            color: {TEXT_COLOR};
        }}
        
        .stSidebar .sidebar-content {{
            background-color: {BACKGROUND_COLOR};
        }}
        
        .stSelectbox>div>div>div {{
            background-color: white;
            border-color: {PRIMARY_COLOR};
            color: {TEXT_COLOR};
        }}
        
        /* Style for dropdown items */
        .stSelectbox div[data-baseweb="select"] > div {{
            color: {TEXT_COLOR};
        }}
        
        .decoration-line {{
            height: 4px;
            background: linear-gradient(90deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
            margin-bottom: 1rem;
        }}
        
        .header-container {{
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }}
        
        .header-line {{
            height: 40px;
            width: 4px;
            background-color: {PRIMARY_COLOR};
            margin-right: 1rem;
        }}
        
        .metrics-container {{
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            color: {TEXT_COLOR};
        }}
        
        .model-info-container {{
            background-color: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 1.5rem;
            color: {TEXT_COLOR};
        }}
        
        .footer {{
            text-align: center;
            margin-top: 2rem;
            color: {TEXT_COLOR};
            font-size: 0.8rem;
        }}
        
        /* Purple zigzag decoration similar to the slide */
        .zigzag-decoration {{
            height: 10px;
            background: linear-gradient(45deg, {PRIMARY_COLOR} 25%, transparent 25%) -5px 0,
                        linear-gradient(-45deg, {PRIMARY_COLOR} 25%, transparent 25%) -5px 0,
                        linear-gradient(45deg, transparent 75%, {PRIMARY_COLOR} 75%),
                        linear-gradient(-45deg, transparent 75%, {PRIMARY_COLOR} 75%);
            background-size: 10px 10px;
            margin: 1rem 0;
        }}
    </style>
    """, unsafe_allow_html=True)

# --- Streamlit App ---
def main():
    set_custom_theme()
    
    # Header with decoration similar to the slide
    st.markdown('<div class="zigzag-decoration"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="header-container">
        <div class="header-line"></div>
        <h1 style="color:#412e3f;">Large Language Models as Cryptanalysts</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<h3 style='color: #412e3f;'>Assessing Decryption Capabilities Across Classical Ciphers</h3>", unsafe_allow_html=True)
    st.markdown('<div class="decoration-line"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: #412e3f;">This demo showcases transformer models trained to decrypt various classical cipher types.
    Select a cipher type, enter your ciphertext, and see the decrypted result.</p>
    """, unsafe_allow_html=True)
    
    # Sidebar with model selection and styling
    with st.sidebar:
        st.markdown(f'<h3 style="color:{PRIMARY_COLOR};">Model Selection</h3>', unsafe_allow_html=True)
        st.markdown('<div class="decoration-line"></div>', unsafe_allow_html=True)
        
        selected_model = st.selectbox(
            "Choose Cipher Type",
            list(MODEL_CONFIGS.keys()),
            index=0
        )
        
        # Apply custom CSS to style the selectbox text
        st.markdown("""
        <style>
        .stSelectbox div[data-baseweb="select"] > div {
            color: #412e3f;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="color: #412e3f;">
        <h3>About</h3>
        This application demonstrates how transformer models can be used for cryptanalysis of classical ciphers.  
        The models have been trained on various cipher types with different complexity levels.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="decoration-line"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="color: #412e3f;">
        <strong>Spring 2025</strong><br>
        American University of Beirut<br>
        Course: CMPS396AH<br>
        Instructor: Amer Mouawad
        </div>
        """, unsafe_allow_html=True)
    
    config = MODEL_CONFIGS[selected_model]
    
    # Load model (with caching)
    @st.cache_resource
    def load_selected_model(model_name):
        model_path = download_model(config["drive_id"], model_name)
        if config["type"] == "vanilla" or config["type"] == "rail_fence":
            return load_vanilla_model(config, model_path)
        elif config["type"] == "enhanced_caesar":
            return load_enhanced_caesar_model(config, model_path)
    
    model, vocab = load_selected_model(selected_model)
    
    # Main content area with styled columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<h3 style="color:{PRIMARY_COLOR};">Input Ciphertext</h3>', unsafe_allow_html=True)
        ciphertext = st.text_area("Enter ciphertext to decrypt:", height=200)
        
        decrypt_button = st.button("Decrypt")
        
        if decrypt_button:
            if not ciphertext.strip():
                st.warning("Please enter some ciphertext to decrypt")
            else:
                with st.spinner(f"Decrypting using {selected_model}..."):
                    try:
                        if config["type"] == "vanilla" or config["type"] == "rail_fence":
                            decrypted = vanilla_decrypt_text(model, ciphertext, vocab, config["max_length"], device)
                        elif config["type"] == "enhanced_caesar":
                            decrypted = enhanced_caesar_decrypt_text(model, ciphertext, vocab, config["max_length"], device)
                        
                        with col2:
                            st.markdown(f'<h3 style="color:{PRIMARY_COLOR};">Decrypted Result</h3>', unsafe_allow_html=True)
                            st.text_area("Decrypted text:", value=decrypted, height=200)
                            
                            # Calculate some metrics with styled container
                            input_len = len(ciphertext)
                            output_len = len(decrypted)
                            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="color: #412e3f;">
                            <strong>Metrics:</strong>
                            <ul>
                                <li>Input length: {input_len} characters</li>
                                <li>Output length: {output_len} characters</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"Error during decryption: {str(e)}")
    
    with col2:
        if not decrypt_button or not ciphertext:
            st.markdown(f'<h3 style="color:{PRIMARY_COLOR};">Decrypted Result</h3>', unsafe_allow_html=True)
            st.info("Enter ciphertext and click 'Decrypt' to see results here")
    
    # Model information section with styled container
    st.markdown('<div class="model-info-container">', unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{PRIMARY_COLOR};">Model Information</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="color: #412e3f;">
        <strong>Selected Model:</strong> {selected_model}
        
        <strong>Architecture Details:</strong>
        <ul>
            <li>Model Type: {'Enhanced Caesar Transformer' if config['type'] == 'enhanced_caesar' else 'Standard Transformer'}</li>
            <li>d_model: {config['d_model']}</li>
            <li>Number of Heads: {config['num_heads']}</li>
            <li>Number of Layers: {config['num_layers']}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="color: #412e3f;">
        <strong>Additional Parameters:</strong>
        <ul>
            <li>Feed Forward Dimension: {config['d_ff']}</li>
            <li>Dropout Rate: {config['dropout']}</li>
            <li>Max Sequence Length: {config['max_length']}</li>
        </ul>
        
        <strong>Performance Note:</strong>
        Character-level accuracy is typically 85-95% for matching cipher types
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("<div style='color: #412e3f;'>Aline Hassan | Zeinab Saad | Hadi Tfaily</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()