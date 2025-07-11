import numpy as np

# 1. Generate Input RGB Values
np.random.seed(42) # for reproducibility
input_rgb = np.random.rand(10, 10, 3) # 10x10 image with 3 color channels

# 2. Flatten the RGB Channels
height, width, channels = input_rgb.shape
flattened_rgb = input_rgb.reshape(height * width, channels)

# Define dimensions (example values)
input_dim = channels
hidden_dim = 64 # Example hidden dimension for Q, K, V

# Initialize random weights for Q, K, V
W_q = np.random.rand(input_dim, hidden_dim)
W_k = np.random.rand(input_dim, hidden_dim)
W_v = np.random.rand(input_dim, hidden_dim)

# Compute Q, K, V
Q = np.dot(flattened_rgb, W_q)
K = np.dot(flattened_rgb, W_k)
V = np.dot(flattened_rgb, W_v)

# 3. Compute Attention Weights
# Scaled dot-product attention
attention_scores = np.dot(Q, K.T) / np.sqrt(hidden_dim)

# 4. Apply Softmax
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(attention_scores)

# 5. Compute Weighted Average
attention_output = np.dot(attention_weights, V)

# 6. Reconstruct Output
# Learnable output weight and bias
W_o = np.random.rand(hidden_dim, channels)
b_o = np.random.rand(channels)

# Apply output weight and add bias (simple linear transformation)
output_transformed = np.dot(attention_output, W_o) + b_o

# Add skip connection (adding original pixel information)
output_rgb_flat = output_transformed + flattened_rgb

# Reshape back to original image dimensions
output_rgb = output_rgb_flat.reshape(height, width, channels)

print("Original RGB shape:", input_rgb.shape)
print("Flattened RGB shape:", flattened_rgb.shape)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)
print("Attention scores shape:", attention_scores.shape)
print("Attention weights shape:", attention_weights.shape)
print("Attention output shape:", attention_output.shape)
print("Output RGB shape:", output_rgb.shape)