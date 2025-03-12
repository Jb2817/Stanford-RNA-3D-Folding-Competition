# Modular Transformer Pipeline for RNA Structure Prediction Using BPPM Tokens
RNA Structure Transformer (RST) is a Python program designed to predict RNA three-dimensional structures by leveraging a combination of tokenized RNA sequences and Base Pair Probability Matrices (BPPMs). The model utilizes a modular transformer architecture, integrating the BPPM with sequence tokens to provide rich contextual information for structure prediction. This dual-input design not only boosts performance by incorporating both primary and secondary structural information but also offers a scalable framework for future enhancements.
It is a solution to this [Kaggle Competition](https://www.kaggle.com/competitions/stanford-rna-3d-folding/overview)

# How to Run
Prepare Your Data

Sequences: Ensure you have a CSV file (e.g., train_sequences.csv) containing RNA sequences along with unique target IDs. RST filters these sequences to include only those composed solely of the canonical nucleotides (A, C, G, U).

BPPM Computation: BPPMs are computed using a C++ Linear Partition project (as described by Zheng et al. in [Bioinformatics](https://academic.oup.com/bioinformatics/article/36/Supplement_1/i258/5870487). The output is a combined text file (e.g., combined_output.txt), where each BPPM entry is separated by an empty line.

Labels (for Training):
For supervised learning, prepare a CSV file (e.g., train_labels.csv) with experimental structure information per residue. This CSV should include columns such as ID, resname, resid, x_1, y_1, z_1, etc.

Place all required files in one folder:
- train_sequences.csv
- train_labels.csv
- combined_output.txt
- BPPM_Transformer.py

The program will load, tokenize, and pad your sequences and lazily process the BPPM matrices to manage RAM usage, then train the transformer model.

# Feature Generation and Model Architecture
# Dual-Input Design
Tokenized RNA Sequences:
Each RNA sequence is tokenized using a predefined vocabulary and padded (or truncated) to a fixed length (e.g., 4300 tokens). This tokenized representation captures the primary sequence information.

BPPM Integration:
The secondary structure of each RNA is represented by a BPPM matrix. The BPPM is computed externally using a C++ Linear Partition project. Because each matrix is padded to 4300×4300, lazy loading is implemented to process only the current sample or batch, alleviating excessive RAM usage.

# Technical Details
PyTorch Framework:
RST is built using PyTorch. Custom Dataset and DataLoader classes ensure efficient batching of tokenized sequences and BPPM matrices. The transformer model itself is implemented using PyTorch’s nn.TransformerEncoderLayer and nn.TransformerEncoder.

Positional Encoding:
Positional encoding is added to the token embeddings to preserve sequential order, which is critical for modeling RNA structure.

Attention Biasing:
The BPPM is incorporated into the multi-head self-attention mechanism as an additive bias. By repeating the BPPM mask across the number of attention heads, the model focuses on residue pairs with high pairing probabilities, enhancing its ability to capture complex structural interactions.

Coordinate Prediction:
The transformer outputs per-residue predictions (x, y, z coordinates) of the C1′ atom. This per-token prediction forms the basis for reconstructing the RNA’s three-dimensional structure.

Model Training
Supervised Learning:
The training dataset pairs each tokenized sequence with its corresponding BPPM and experimental structure data from train_labels.csv. Ground-truth coordinates serve as targets for learning.

Loss Function:
The model is trained using Mean Squared Error (MSE) loss to minimize the difference between predicted and experimental coordinates.

Batching and Optimization:
A custom PyTorch DataLoader batches both the tokenized sequences and BPPM matrices, ensuring efficient memory use. Given the high memory demand (each BPPM matrix is huge), lazy loading is used so that only the required matrices are loaded on demand.

Inference and Submission
Test Data Processing:
For inference, the same preprocessing pipeline is applied (excluding labels). New sequences and their corresponding BPPMs are processed identically to the training data, ensuring consistency.

CSV Output:
The model’s per-residue coordinate predictions are formatted into a CSV file (e.g., sample_submission.csv) as per competition guidelines. Each row includes:

ID: A combination of the target ID and residue number (using one-based indexing)
resname: The RNA nucleotide
resid: Residue number
Coordinates: Predicted x, y, and z values
(Currently, the model outputs one set of coordinates per residue; future iterations will extend this to output five sets for the final submission.)
