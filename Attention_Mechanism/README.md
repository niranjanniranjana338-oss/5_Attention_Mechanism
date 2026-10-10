# Week 5 – Attention Mechanism

## Project Overview
This project demonstrates the Scaled Dot-Product Attention mechanism used in Transformer models and Large Language Models (LLMs).

## Objectives
- Understand the attention mechanism.
- Create Query, Key, and Value matrices.
- Calculate attention scores and attention weights.
- Generate the attention output.
- Visualize attention weights using a heatmap.
- Compare NumPy and PyTorch implementations.

## Technologies Used
- Python
- NumPy
- PyTorch
- Matplotlib
- Jupyter Notebook

## Project Structure
- dataset/: Contains sample sentences, attention weights, attention output, and heatmap.
- notebook/: Contains the Jupyter Notebook.
- attention.py: Python implementation of attention.
- requirements.txt: Required Python libraries.

## How to Run
1. Install the required libraries:
   python -m pip install -r requirements.txt

2. Run the Python program:
   python attention.py

3. Open the notebook:
   notebook/Week5_Attention_Mechanism.ipynb

## Results
The program calculates attention weights and attention outputs. The NumPy and PyTorch implementations produce matching results.

## Conclusion
This project helped demonstrate how the attention mechanism calculates relationships between input vectors and produces weighted output representations.

## Observations

1. Attention weights show how much attention each query gives to different key tokens.
2. The softmax function converts attention scores into weights between 0 and 1.
3. The attention weights for each query sum to approximately 1.
4. The attention output is calculated by multiplying the attention weights with the Value matrix.
5. The heatmap provides a visual representation of the attention weights.
6. NumPy and PyTorch produce matching attention results in this implementation.

## Applications of Attention

- Attention is an important component of Transformer models.
- It helps language models process relationships between tokens.
- It is used in applications such as text generation, translation, and question answering.