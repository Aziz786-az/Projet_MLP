"""Le classifieur C : un perceptron à une couche cachée."""
import torch
import torch.nn as nn


class MLPLangModel(nn.Module):

    def __init__(self, vocab_size, emb_dim, k, dh, emb_init=None, freeze_emb=True):
        super().__init__()
        self.k, self.emb_dim = k, emb_dim
        self.emb = nn.Embedding(vocab_size, emb_dim)           
        if emb_init is not None:
            with torch.no_grad():
                self.emb.weight.copy_(emb_init)
        self.emb.weight.requires_grad = not freeze_emb         
        self.hidden = nn.Linear(k * emb_dim, dh)               
        self.out = nn.Linear(dh, vocab_size)                   

    def forward(self, x):                      
        e = self.emb(x)                        
        e = e.reshape(e.size(0), -1)           
        h = torch.relu(self.hidden(e))        
        return self.out(h)                     

    @torch.no_grad()
    def log2_probs(self, x):
        """log2 P(. | contexte) pour chaque ligne de x."""
        return torch.log_softmax(self(x), dim=-1) / torch.log(torch.tensor(2.0))
