"""Étapes 2 et 3 du TP : boucle d'apprentissage, early stopping, sauvegarde.

Modèle de base (hyperparamètres du sujet) :
    python train.py --train train_12k.bpe --val val_12k.bpe --emb_file bpe_embeddings_12k_32.txt

Piste 1 (embeddings appris) :
    python train.py ... --init random --train_emb --out resultats/mlp_emb_random.pt
Piste 3 (embeddings appris, initialisés avec SGNS) :
    python train.py ... --init_from resultats/sgns_32.txt --train_emb --out resultats/mlp_emb_sgns.pt
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

from common import cached_data, get_device, load_vocab
from model import MLPLangModel


def load_init_embeddings(path, vocab):
    """Charge un autre fichier d'embeddings (ex. SGNS du TP1) aligné sur le vocabulaire
    du modèle. Les tokens absents gardent un vecteur aléatoire."""
    other = load_vocab(path)
    if other.emb_dim != vocab.emb_dim:
        raise ValueError(f"dimension {other.emb_dim} != {vocab.emb_dim}")
    E = torch.randn(len(vocab.word_array), vocab.emb_dim) * 0.1
    found = 0
    for tok, i in vocab.dico_voca.items():
        j = other.dico_voca.get(tok)
        if j is not None:
            E[i] = other.matrice[j]
            found += 1
    print(f"init depuis {path} : {found}/{len(vocab.word_array)} tokens trouvés")
    return E


@torch.no_grad()
def evaluate(model, X, Y, device, batch_size=4096):
    """Perte moyenne (en nats) sur (X, Y)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    for i in range(0, len(Y), batch_size):
        x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
        total += loss_fn(model(x), y).item()
    return total / len(Y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--emb_file", required=True, help="fichier bpe_embeddings_12k_n.txt (définit le vocabulaire)")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--dh", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=2, help="early stopping")
    p.add_argument("--init", choices=["file", "random"], default="file",
                   help="file = embeddings du fichier ; random = aléatoires")
    p.add_argument("--init_from", default=None, help="autre fichier d'embeddings pour l'init (piste 3)")
    p.add_argument("--train_emb", action="store_true", help="embeddings mis à jour (piste 1)")
    p.add_argument("--out", default="resultats/mlp.pt")
    p.add_argument("--max_lines", type=int, default=None, help="limiter le corpus (débogage)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    print("device :", device)

    vocab = load_vocab(args.emb_file)
    V, d = vocab.matrice.shape

    X_tr, Y_tr = cached_data(args.train, vocab, args.k, args.max_lines)
    X_va, Y_va = cached_data(args.val, vocab, args.k, args.max_lines)

    if args.init_from:
        E0 = load_init_embeddings(args.init_from, vocab)
    elif args.init == "file":
        E0 = vocab.matrice
    else:
        E0 = None                                   # initialisation aléatoire de nn.Embedding
    model = MLPLangModel(V, d, args.k, args.dh, emb_init=E0, freeze_emb=not args.train_emb).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"|V|={V} d={d} k={args.k} dh={args.dh} | paramètres entraînables : {n_params:,}")

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    config = dict(emb_file=args.emb_file, vocab_size=V, emb_dim=d, k=args.k, dh=args.dh,
                  lr=args.lr, batch_size=args.batch_size, init=args.init,
                  init_from=args.init_from, train_emb=args.train_emb, seed=args.seed)
    history = []
    best_val, bad_epochs = float("inf"), 0

    # perte de départ (utile pour comparer les vitesses d'apprentissage, piste 3)
    v0 = evaluate(model, X_va, Y_va, device)
    print(f"epoch 0 | val {v0 / math.log(2):.4f} bits (PP {math.exp(v0):.2f})")
    history.append(dict(epoch=0, train=None, val=v0))

    N = len(Y_tr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(N)                    # mélange des exemples à chaque epoch
        total = 0.0
        n_batches = (N + args.batch_size - 1) // args.batch_size
        report_every = max(1, n_batches // 20)       # affiche la progression tous les 5 %
        for b, i in enumerate(range(0, N, args.batch_size)):
            if b and b % report_every == 0:
                el = time.time() - t0
                print(f"  epoch {epoch} : {100 * b / n_batches:3.0f} % | perte {total / (i * math.log(2)):.3f} bits "
                      f"| reste ~{el * (n_batches - b) / b / 60:.1f} min", flush=True)
            idx = perm[i:i + args.batch_size]
            x, y = X_tr[idx].to(device), Y_tr[idx].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)             # graphe de calcul de la perte
            loss.backward()                         # gradient
            optimizer.step()                        # pas d'optimisation
            total += loss.item() * len(idx)

        train_loss = total / N
        val_loss = evaluate(model, X_va, Y_va, device)
        history.append(dict(epoch=epoch, train=train_loss, val=val_loss))
        print(f"epoch {epoch} | train {train_loss / math.log(2):.4f} bits | "
              f"val {val_loss / math.log(2):.4f} bits (PP {math.exp(val_loss):.2f}) | "
              f"{time.time() - t0:.0f}s")

        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            torch.save({"config": config, "state": model.state_dict(), "history": history}, args.out)
            print(f"  -> meilleur modèle sauvegardé dans {args.out}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping (pas d'amélioration depuis {args.patience} epochs)")
                break

    # on garde l'historique complet dans la sauvegarde et dans un json (courbes du rapport)
    ckpt = torch.load(args.out, map_location="cpu")
    ckpt["history"] = history
    torch.save(ckpt, args.out)
    with open(args.out.rsplit(".", 1)[0] + "_history.json", "w") as f:
        json.dump({"config": config, "history": history}, f, indent=1)
    print(f"meilleure perplexité validation : {math.exp(best_val):.2f}")


if __name__ == "__main__":
    main()
