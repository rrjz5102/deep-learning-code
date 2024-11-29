import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{original_repr(self)}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device to GPU if available, otherwise CPU



def show_heatmaps(attention_weight):
    if type(attention_weight) != np.array:
        attention_weight = np.array(attention_weight)
    plt.imshow(attention_weight, cmap='Reds', vmin=0, vmax=1)  # Set cmap to 'Reds' and define vmin/vmax
    plt.colorbar()  # Add colorbar for reference
    plt.title('Attention Heatmap')  # Add title for clarity
    plt.xlabel('Keys')  # Label for x-axis
    plt.ylabel('Queries')  # Label for y-axis
    plt.show()
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings,embedding_dim):
        super().__init__()
        # ignore <pad> token
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=1)
        
    def forward(self, x):
        # x should be int or longint
        return self.embedding(x)
          
class PositionalEmbedding(nn.Module):
    """
    Positional Embedding for Transformer models.
    
    Args:
        max_len (int): Maximum length of the input sequences.
        d_model (int): Dimensionality of the embeddings.
    """
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros((max_len, d_model), requires_grad=False)

        # sin: [0, 2, 4, 6,...], cos: [1, 3, 5, 7,...]
        _2i = torch.arange(0, d_model, 2)  # Even indices
        pos = torch.arange(max_len).unsqueeze(1)  # Shape: (max_len, 1)

        # Apply sine to even indices
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        # Apply cosine to odd indices
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        # x: (batch_size, seq_len)
        return self.encoding[:x.size(1), :]# Shape: (1, seq_len, d_model) for broadcasting

class TransformerEmbedding(nn.Module):
    """
        word embedding + position embedding
    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, vocab_size,d_model, max_len, drop_prob):
        super().__init__()
        self.emb = Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(max_len, d_model)
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, x):
        token_emb = self.emb(x)   # (batch, vocab_size, d_model)
        pos_emb = self.pos_emb(x) # (1, seq_len, d_model) 
        
        return self.dropout(token_emb + pos_emb)
        

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask (if any)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Compute the output
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        """_summary_

        Args:
            d_model (_type_): _description_
            num_heads (_type_): _description_
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # project Q, K and V using linear layer and then split to h parts
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # linear project for attention_weight
        self.linear = nn.Linear(d_model, d_model)
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # split to h heads (batch_size, num_heads, N, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        
        # compute multihead attention
        score = torch.matmul(Q,K.transpose(-2,-1)) / (self.d_k ** 0.5)  #(B, num_heads,N,N)
        
        # padding mask or decoding mask
        if mask is not None:
            score = score.masked_fill(mask==0, float('-inf'))
        
        attention_weight = F.softmax(score, dim=-1) #(B, num_heads,N,N)
        output = attention_weight @ V  #(B, num_heads,N,d_k)
        
        # concact and transpose to the original shape
        output = output.transpose(1, 2).contiguous()  # Change order back to (batch_size, N, num_heads, d_k)
        output = output.view(batch_size, -1, self.num_heads * self.d_k)  # Combine heads
        output = self.linear(output).to(device)
        
        return output, attention_weight
    
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-10):
        super().__init__()
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x):
        # x: (batch, seq_len,d_model)
        mean = x.mean(dim=2, keepdim=True)
        var = torch.mean((x - mean)**2, dim=-1, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
        
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_ffn=2048, d_model=512, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout_rate)
   
    def forward(self, x):
        # x: (B, seq_len, d_model)
        out = self.fc1(x)
        out = F.relu(out)  #max(0,x)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
        
class EncoderLayer(nn.Module):
    
    """
        Implement the encoder layer
    Args:
        nn (_type_): _description_
    """
    def __init__(self, d_model=512, d_ffn=2048, n_head=8, drop_prob=0.2):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.layernorm1 = LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_ffn, d_model, drop_prob)
        self.layernorm2 = LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """_summary_

        Args:
            x (_type_): (batch, seq_len, d_model)
            mask (_type_, optional): _description_. Defaults to None.
        """
        # sublayer 1
        out1, attention_weight= self.attention(x,x,x, mask)
        out1 = self.layernorm1(x+out1)
        
        # sublayer 2
        out2 = self.ffn(out1)
        out2 = self.layernorm2(out1 + out2)
        
        return out2, attention_weight
        
class Encoder(nn.Module):
    def __init__(self, max_len=512,n_stack=6,d_model=512, d_ffn=2048, n_head=8, drop_prob=0.2):
        super().__init__()
        self.n_stack = n_stack
        #! Must be ModuleList. If it is a list, it wont be moved to gpu
        self.encoder = nn.Sequential(*[
                EncoderLayer(d_model, d_ffn, n_head, drop_prob) for _ in range(n_stack)
             ])        
    def forward(self, x, mask=None):
        #TODO Embedding
        for i in range(self.n_stack):
            x, attention_weight = self.encoder[i](x, mask)
        return x, attention_weight
         
        
                             
if __name__ =='__main__':
    batch_size = 2
    N = 10
    d_model = 512
 
    
    x = torch.rand(batch_size, N, d_model).to(device)
    x = x.long()
    #query = torch.rand(batch_size, N, d_model)  # Shape: (batch_size, seq_len_q, d_model)
    #key = torch.rand(batch_size, N, d_model)    # Shape: (batch_size, seq_len_k, d_model)
    #value = torch.rand(batch_size, N, d_model)  # Shape: (batch_size, seq_len_k, d_model)
    mask=None
    # att = DotProductAttention()
    # attention_weights,output = att(query, key, value)
    #multiHeadAttention = MultiHeadAttention()
    #output, attention_weight = multiHeadAttention(query, key, value)
    #print(output.shape, attention_weight.shape)
    #print(multiHeadAttention.d_k)
    #layer = LayerNorm(d_model)
    #layer = EncoderLayer(d_model=512, d_ffn=2048, n_head=8, drop_prob=0.2).to(device)
    #layer = Encoder(max_len=512,n_stack=6,d_model=512, d_ffn=2048, n_head=8, drop_prob=0.2).to(device)
    layer = TransformerEmbedding(vocab_size=100, d_model=512, max_len=10, drop_prob=0.2).to(device)
    # Check devices
 
    output = layer(x)
    #print(x.shape)
