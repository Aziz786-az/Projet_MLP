"""Embeddings Skip-Gram with Negative Sampling (SGNS) sur le corpus BPE.

    python sgns.py --corpus train_12k.bpe --emb_file bpe_embeddings_12k_32.txt    -> resultats/sgns_32.txt

Réglages par défaut = configuration « all » du TP1 (la meilleure) :
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from common import get_device, load_vocab, sentence_tokens, token_to_id


def load_corpus(path, vocab, max_lines=None):
    """Indices des tokens + numéro de phrase de chaque token (sans <s>/</s> de remplissage)."""
    ids, sent = [], []
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_lines is not None and n >= max_lines:
                break
            toks = sentence_tokens(line, 1)        # un seul <s> et un seul </s> par phrase
            if toks is None:
                continue
            line_ids = [token_to_id(vocab, t) for t in toks]
            if any(i is None for i in line_ids):
                continue
            ids.extend(line_ids)
            sent.extend([n] * len(line_ids))
            n += 1
    print(f"{path} : {n} phrases, {len(ids)} tokens")
    return torch.tensor(ids), torch.tensor(sent)


def make_pairs(ids, sent, L):
    """Toutes les paires (cible, contexte) à distance <= L, dans la même phrase."""
    b = torch.randint(1, L + 1, (len(ids),))
    targets, contexts = [], []
    for o in range(1, L + 1):
        same = sent[:-o] == sent[o:]
        left, right = ids[:-o], ids[o:]
        fwd = same & (b[:-o] >= o)                  # cible à gauche, contexte à droite
        bwd = same & (b[o:] >= o)                   # cible à droite, contexte à gauche
        targets += [left[fwd], right[bwd]]
        contexts += [right[fwd], left[bwd]]
    return torch.cat(targets), torch.cat(contexts)


def nearest(E, vocab, words, n=8):
    En = F.normalize(E, dim=1)
    for w in words:
        i = vocab.dico_voca.get(w)
        if i is None:
            continue
        sims = En @ En[i]
        best = sims.topk(n + 1).indices.tolist()[1:]
        print(f"  {w:>10} : " + " ".join(vocab.word_array[j] for j in best))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--emb_file", required=True, help="fichier fourni (définit le vocabulaire et l'ordre des tokens)")
    p.add_argument("--out", default=None, help="par défaut resultats/sgns_<dim>.txt")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--window", type=int, default=2, help="L : taille de la fenêtre de chaque côté")
    p.add_argument("--neg", type=int, default=10, help="k : nombre d'exemples négatifs par paire")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-2,
                   help="taux initial (Adam), décroissant linéairement jusqu'à lr * 1e-4 (comme au TP1)")
    p.add_argument("--max_lines", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if args.out is None:
        args.out = f"resultats/sgns_{args.dim}.txt"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    print("device :", device)

    vocab = load_vocab(args.emb_file)
    V = len(vocab.word_array)
    ids, sent = load_corpus(args.corpus, vocab, args.max_lines)

    # distribution des négatifs : fréquence ^ 0.75
    counts = torch.bincount(ids, minlength=V).float()
    neg_dist = (counts ** 0.75)
    neg_dist = (neg_dist / neg_dist.sum()).to(device)

    freq = counts / counts.sum()
    keep_prob = ((1e-3 / freq.clamp(min=1e-12)).sqrt()).clamp(max=1.0)

    E = torch.nn.Embedding(V, args.dim).to(device)     
    C = torch.nn.Embedding(V, args.dim).to(device)     
    torch.nn.init.uniform_(E.weight, -0.5 / args.dim, 0.5 / args.dim)
    torch.nn.init.uniform_(C.weight, -0.5 / args.dim, 0.5 / args.dim)
    opt = torch.optim.Adam(list(E.parameters()) + list(C.parameters()), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        # nouveau sous-échantillonnage à chaque epoch, puis construction des paires
        keep = torch.rand(len(ids)) < keep_prob[ids]
        tg, ct = make_pairs(ids[keep], sent[keep], args.window)
        perm = torch.randperm(len(tg))
        tg, ct = tg[perm], ct[perm]
        n_batches = math.ceil(len(tg) / args.batch_size)
        print(f"epoch {epoch} : {keep.float().mean():.0%} des tokens gardés, {len(tg):,} paires")

        t0, total = time.time(), 0.0
        for b in range(n_batches):
            progress = (epoch - 1 + b / n_batches) / args.epochs
            for g in opt.param_groups:
                g["lr"] = max(args.lr * (1.0 - progress), args.lr * 1e-4)
            t = tg[b * args.batch_size:(b + 1) * args.batch_size].to(device)
            c = ct[b * args.batch_size:(b + 1) * args.batch_size].to(device)
            n = torch.multinomial(neg_dist, len(t) * args.neg, replacement=True).view(len(t), args.neg)

            e = E(t)                                              
            pos = (e * C(c)).sum(-1)                              
            negs = torch.bmm(C(n), e.unsqueeze(-1)).squeeze(-1)   
            loss = -(F.logsigmoid(pos) + F.logsigmoid(-negs).sum(-1)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            if (b + 1) % max(1, n_batches // 10) == 0:
                print(f"  {100 * (b + 1) / n_batches:3.0f} % | perte {total / (b + 1):.4f} "
                      f"| {time.time() - t0:.0f}s", flush=True)

        print(f"epoch {epoch} | perte moyenne {total / n_batches:.4f}")
        nearest(E.weight.detach().cpu(), vocab, ["il", "maison", "rouge", "dit", "deux", "paris"])

    W = E.weight.detach().cpu()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"{V} {args.dim}\n")
        for i, tok in enumerate(vocab.word_array):
            f.write(tok + " " + " ".join(f"{x:.6f}" for x in W[i].tolist()) + "\n")
    print(f"embeddings SGNS écrits dans {args.out} ({V} tokens, dimension {args.dim})")


if __name__ == "__main__":
    main()
