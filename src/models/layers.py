import torch
from torch import nn
from torch.nn import functional as F
import math
from einops import einsum, rearrange


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class RotaryPositionalEmbeddings(nn.Module):
    """ 
    Implementation of RoPE introduced in the paper RoFormer: Enhanced Transformer with Rotary Position Embedding.
    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, d: int, base: int = 10_000):
        super().__init__()

        self.d = d
        self.base = base
        self.cos_cache = torch.empty(0)
        self.sin_cache = torch.empty(0)

    

    def _build_cache(self, x: torch.Tensor):
        """
        Build a cache for efficient computation of the rotary embeddings.
        """
    
        b, h, t, n_embd = x.size() # batch, n_heads, seq_len, embed_dim per head

        assert (self.d == n_embd)
        assert (n_embd % 2 == 0)
        
        i = torch.arange(1, self.d//2 + 1, device=x.device)  # i values from 1 to d/2
        Theta = self.base ** (-2 * (i - 1) / self.d) # d/2 theta values
        Theta = Theta.unsqueeze(0) # [1, d/2]

        j = torch.arange(1, t+1, device=x.device).unsqueeze(1) # [t, 1]

        C = j * Theta # [t, d/2]
        C = torch.cat([C, C], dim=1) # [t, d]

        cos_matrix = torch.cos(C) # [t, d]
        sin_matrix = torch.sin(C) # [t, d]

        # need cosine and sine matrix for each head:
        cos_matrix = cos_matrix.unsqueeze(1).expand(-1, h, -1) # [t, h, d]
        sin_matrix = sin_matrix.unsqueeze(1).expand(-1, h, -1) # [t, h, d]

        self.cos_cache = rearrange(cos_matrix, 't h d -> h t d')
        self.sin_cache = rearrange(sin_matrix, 't h d -> h t d')

        # push to device
        self.cos_cache = self.cos_cache.to(x.device)
        self.sin_cache = self.sin_cache.to(x.device)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_query_head == 0
        self.n_head = config.n_query_head
        self.n_embd = config.n_embd
        self.ROPE = None

        # key, query, value projections
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # output projection
        self.out = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

        self.rope = config.rope
        if self.rope:
            self.ROPE = RotaryPositionalEmbeddings(self.n_embd // self.n_head)


    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        B, T, C = q.size() # batch size, sequence length, embedding dimensionality (n_embd)


        # split the embedding dimension (n_embd) across the number of heads
        # reshape the query, key, value tensors to increase efficiency of matrix multiplication
        # b = batch size, t = sequence length, h = number of heads, d = n_embd / number of heads
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_head)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_head)
        
        if self.rope:
            q = self.ROPE(q)
            k = self.ROPE(k)
        
        # compute square root of (n_embd / number of heads) to scale the dot product
        scale = math.sqrt(k.size(-1))
        
        # calculate the attention scores with the query and key
        att = einsum(q, k, 'b h q d, b h k d -> b h q k') / scale
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att_weights = self.attn_dropout(att)
        
        # matrix multiplication of attention scores and value
        y = einsum(att, v, 'b h q t, b h t d -> b h q d')
        
        # rearrange the output tensor to (batch size, sequence length, n_embd)
        y = rearrange(y, 'b h q d -> b q (h d)') # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.out(y))
        
        return y, att_weights

class GroupedQueryAttention(nn.Module):
    """
    An implementation of group query attention. Refer to the CausalSelfAttention class to structure your implementation.
    """

    def __init__(self, config):
        super().__init__()

        """
        Implementation of grouped query attention
        """

        assert config.n_embd % config.n_query_head == 0
        assert config.n_embd % config.n_kv_head == 0
        assert config.n_query_head % config.n_kv_head == 0

        self.n_query_head = config.n_query_head
        self.n_kv_head = config.n_kv_head
        self.group_size = self.n_query_head // self.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_query_head
        self.ROPE = None

        # Key, Query, Value Projections
        self.query = nn.Linear(self.n_embd, self.n_embd) 
        self.key = nn.Linear(self.n_embd, self.head_dim * self.n_kv_head) 
        self.value = nn.Linear(self.n_embd, self.head_dim * self.n_kv_head) 

        # Output Projection
        self.out = nn.Linear(self.head_dim * self.n_query_head, self.n_embd)  

        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        
        self.rope = config.rope
        if self.rope:
            self.ROPE = RotaryPositionalEmbeddings(self.head_dim)
            

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        B, T, C = q.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # split across heads by introducing an additional 'h' dimension
        # reshape the query, key, value tensors to increase efficiency of matrix multiplication
        # b = batch size, t = sequence length, h = number of heads, dk
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_query_head)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_kv_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_kv_head)
        
        if self.rope:
            q = self.ROPE(q)
            k = self.ROPE(k)        

        q = rearrange(q, 'b (h g) t d -> b g h t d', g=self.group_size)
        
        # compute square root of (n_embd / number of heads) to scale the dot product
        scale = math.sqrt(k.size(-1))
        
        # calculate the attention scores with the query and  key
        att = einsum(q, k, 'b g h q d, b h k d -> b g h q k') / scale

        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att_weights = self.attn_dropout(att)

        # matrix multiplication of attention scores and value
        y = einsum(att, v, 'b g h q k, b h k d -> b g h q d')

        # rearrange the output tensor to (batch size, sequence length, n_embd)
        y = rearrange(y, 'b g h q d -> b q (h g) d') # re-assemble all head outputs side by side
        y = rearrange(y, 'b q h d -> b q (h d)')
       
        # output projection
        y = self.resid_dropout(self.out(y))

        return y, att_weights
    

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        if config.n_query_head != config.n_kv_head:
            self.attn = GroupedQueryAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        attn_comp, attn_weights = self.attn(self.ln_1(x))
        x = x + attn_comp
        y = x + self.mlpf(self.ln_2(x))
        return y