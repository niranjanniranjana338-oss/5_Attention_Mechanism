import csv
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """Compute attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V."""
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    weights = softmax(scores)
    output = weights @ V
    return output, weights

def main():
    tokens = ['I', 'love', 'AI']
    # Small deterministic toy embeddings, used to demonstrate the calculation.
    X = np.array([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.]])
    Q, K, V = X.copy(), X.copy(), X.copy()
    output, weights = scaled_dot_product_attention(Q, K, V)
    data_dir = BASE / 'dataset'
    with open(data_dir / 'attention_weights.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['query_token'] + tokens)
        for token, row in zip(tokens, weights): writer.writerow([token] + [round(float(x), 6) for x in row])
    with open(data_dir / 'attention_output.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['token', 'output_dim_1', 'output_dim_2', 'output_dim_3'])
        for token, row in zip(tokens, output): writer.writerow([token] + [round(float(x), 6) for x in row])
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(weights, cmap='viridis')
        ax.set_xticks(range(len(tokens)), tokens); ax.set_yticks(range(len(tokens)), tokens)
        ax.set_xlabel('Key tokens'); ax.set_ylabel('Query tokens'); ax.set_title('Scaled Dot-Product Attention')
        fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(data_dir / 'attention_heatmap.png', dpi=150); plt.close(fig)
    except ImportError:
        print('Install matplotlib to generate the heatmap image.')
    print('Attention weights:\n', np.round(weights, 3))
    print('Attention output:\n', np.round(output, 3))
    try:
        import torch
        tq, tk, tv = [torch.tensor(a, dtype=torch.float32) for a in (Q, K, V)]
        tw = torch.softmax(tq @ tk.T / np.sqrt(tk.shape[-1]), dim=-1)
        tout = tw @ tv
        print('PyTorch output matches NumPy:', np.allclose(tout.numpy(), output, atol=1e-6))
    except ImportError:
        print('PyTorch comparison skipped. Install torch to enable it.')

if __name__ == '__main__': main()
