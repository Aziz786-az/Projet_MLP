"""Le classifieur C : un perceptron à une couche cachée."""
import torch
import torch.nn as nn


class MLPLangModel(nn.Module):
    """x = [E[m_{i-k}] ; ... ; E[m_{i-1}]]   (taille k*d)
    h = ReLU(x W + b1)                       W : (k*d) x dh
    y' = h U + b2                            U : dh x |V|
    forward renvoie les logits y' ; la softmax est dans CrossEntropyLoss
    (ou appliquée explicitement en évaluation / génération)."""

    def __init__(self, vocab_size, emb_dim, k, dh, emb_init=None, freeze_emb=True):
        super().__init__()
        self.k, self.emb_dim = k, emb_dim
        self.emb = nn.Embedding(vocab_size, emb_dim)           # matrice E : |V| x d
        if emb_init is not None:
            with torch.no_grad():
                self.emb.weight.copy_(emb_init)
        self.emb.weight.requires_grad = not freeze_emb         # piste 1 : freeze_emb=False
        self.hidden = nn.Linear(k * emb_dim, dh)               # W, b1
        self.out = nn.Linear(dh, vocab_size)                   # U, b2

    def forward(self, x):                      # x : (B, k) indices
        e = self.emb(x)                        # (B, k, d)
        e = e.reshape(e.size(0), -1)           # (B, k*d) : concaténation des k plongements
        h = torch.relu(self.hidden(e))         # (B, dh)
        return self.out(h)                     # (B, |V|)

    @torch.no_grad()
    def log2_probs(self, x):
        """log2 P(. | contexte) pour chaque ligne de x."""
        return torch.log_softmax(self(x), dim=-1) / torch.log(torch.tensor(2.0))
