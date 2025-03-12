import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

######################################
#Informs the model of the sequence/position of tokens
######################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4600):
        """
        Makes the model understand where each part of the sequence is
        
        d_model = size of hidden dimensions
        dropout = prevents overfitting by deactivating random neurons
        
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)] #Adds positional encoding to input
        return self.dropout(x)

######################################
# Configs
######################################
MAX_LENGTH = 4300 #Longest RNA sequence in dataset
BATCH_SIZE = 1  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
OUTPUT_DIM = 3  #X, Y, Z coordinates for the C1 atom
"""
Not sure of the difference between X and - but both appear in multiple sequences and together occasionally 
"""
vocab = {

    'A': 0,
    'C': 1,
    'G': 2,
    'U': 3,
    'X': 4, 
    '-': 5, 
    'PAD': 6  #For equal sized tokens
}
PAD_TOKEN = vocab['PAD']


def tokenize_sequence(seq, vocab=vocab):
    """
    Converts RNA bases to integers
    """
    tokens = []
    for char in seq:
        if char in vocab:
            tokens.append(vocab[char])
        else:
            tokens.append(vocab['X'])
    return tokens

def pad_sequence(tokenized_seq, max_length=MAX_LENGTH, pad_token=PAD_TOKEN):
    """
    Make all sequences the same length by adding padding
    """
    seq_length = len(tokenized_seq)
    if seq_length < max_length:
        padded_seq = tokenized_seq.copy()
        for i in range(max_length - seq_length):
            padded_seq.append(pad_token)
        return padded_seq
    else:
        return tokenized_seq[:max_length]

def tokenize_and_pad(seq, max_length=MAX_LENGTH):
    tokens = tokenize_sequence(seq)
    padded = pad_sequence(tokens, max_length)
    return torch.tensor(padded, dtype=torch.long)

def parse_bppm_file(file_path):
    """
    Read the base pair probability file and add each matrix as an element in bppm_entries
    """
    bppm_entries = []
    current_entry = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            i = int(parts[0])
            j = int(parts[1])
            prob = float(parts[2])
            
            
            if current_entry and i < current_entry[-1][0]: #If a new sequence starts, save the old one
                bppm_entries.append(current_entry)
                current_entry = []
           
            current_entry.append((i, j, prob))
    
    if current_entry:
        bppm_entries.append(current_entry)
    
    return bppm_entries

def pad_bppm_matrix(bppm, max_length=MAX_LENGTH):
    """
    Make all matrices the same size
    """
    n = bppm.shape[0]
    if n < max_length:
        pad_size = max_length - n
        padded = F.pad(bppm, (0, pad_size, 0, pad_size), "constant", 0)
    elif n > max_length:
        padded = bppm[:max_length, :max_length]
    else:
        padded = bppm
    return padded

######################################
# Dataset class for loading RNA sequences and their base pair info
######################################
class RNADataset(Dataset):
    def __init__(self, csv_path, bppm_file_path, seq_col='sequence', max_length=MAX_LENGTH):
        self.df = pd.read_csv(csv_path)
        
        valid_nucleotides = set(['A', 'C', 'G', 'U'])
        filtered_rows = []
        for i, row in self.df.iterrows():
            seq = row[seq_col]
            is_valid = True
            for char in seq:
                if char not in valid_nucleotides:
                    is_valid = False
                    break
            if is_valid:
                filtered_rows.append(row)
        
        self.df = pd.DataFrame(filtered_rows)
        self.sequences = self.df[seq_col].tolist()
        self.max_length = max_length
        
        bppm_entries = parse_bppm_file(bppm_file_path)
        
        if len(bppm_entries) < len(self.sequences): #Not enough BPPMs for number of sequences 
            raise ValueError(f"Not enough BPPM entries: {len(self.sequences)} sequences but only {len(bppm_entries)} entries.")
        
       
        if len(bppm_entries) > len(self.sequences):
            bppm_entries = bppm_entries[:len(self.sequences)]
        
        self.bppm_entries = bppm_entries

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokenized_seq = tokenize_and_pad(seq, max_length=self.max_length)
        entry = self.bppm_entries[idx]
 
        max_idx = 0
        for (i, j, prob) in entry:
            if i > max_idx:
                max_idx = i
            if j > max_idx:
                max_idx = j
        
        matrix = torch.zeros((max_idx, max_idx), dtype=torch.float)
        
        
        for (i, j, prob) in entry:   #Fills in the matrix with probabilities & mirrors across main diagonal
            matrix[i-1, j-1] = prob
            matrix[j-1, i-1] = prob  
        
        bppm_matrix = pad_bppm_matrix(matrix, self.max_length) 
        
        return tokenized_seq, bppm_matrix

######################################
# Transformer model to predict 3D coordinates
######################################
class TransformerForCoordinates(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=2, output_dim=OUTPUT_DIM, max_length=MAX_LENGTH, dropout=0.1):
        super(TransformerForCoordinates, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        #Create transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        #Final layer to predict coordinates
        self.fc = nn.Linear(d_model, output_dim)
        self.d_model = d_model
        self.max_length = max_length
        self.nhead = nhead  

    def forward(self, src, bppm):
        #shapes
        batch_size = src.size(0)
        seq_len = src.size(1)
        
        src = self.embedding(src) * math.sqrt(self.d_model) #Embeds sequence tokens
        src = self.pos_encoder(src) #Adds positional encoder
        
       
        src = src.transpose(0, 1)
        bppm = bppm[:, :seq_len, :seq_len]
        
        #Creates attention bias from BPPM
        attn_bias = bppm.clone().detach()
        attn_bias = attn_bias.masked_fill(attn_bias == 0, float('-inf'))
        
       
        attn_bias = attn_bias.repeat(self.nhead, 1, 1)
        
        #Run transformer layers
        for layer in self.transformer_encoder.layers:
            _, _ = layer.self_attn(src, src, src, attn_mask=attn_bias)
            src = layer(src)
        
        #Transpose back to (batch_size, seq_len, d_model)
        encoded = src.transpose(0, 1)
        
        #Predict coordinates
        coords = self.fc(encoded) 
        return coords

######################################
# Running the model
######################################
if __name__ == '__main__':
    csv_path = 'train_sequences.csv'
    bppm_file_path = 'combined_output.txt'
    
    #Creates Dataset
    dataset = RNADataset(csv_path, bppm_file_path, seq_col='sequence', max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    #Batch Info
    for batch_idx, (token_batch, bppm_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print("  Token batch shape:", token_batch.shape)       
        print("  BPPM batch shape:", bppm_batch.shape)           
        break
    
    
    model = TransformerForCoordinates(vocab_size=len(vocab), d_model=128, nhead=8, num_layers=2, output_dim=OUTPUT_DIM, max_length=MAX_LENGTH, dropout=0.1)
    model.to(DEVICE)
    

    for token_batch, bppm_batch in dataloader:
        token_batch = token_batch.to(DEVICE)
        bppm_batch = bppm_batch.to(DEVICE)
        outputs = model(token_batch, bppm_batch)
        print("Output shape:", outputs.shape)  
        break
