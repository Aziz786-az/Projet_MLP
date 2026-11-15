"""Perplexité d'un texte selon un modèle sauvegardé.
    python perplexite.py mlp.pt test_12k.bpe
"""
import argparse

import torch

from common import EOS, get_device, load_model, prepare_data


@torch.no_grad()
def log2_probs_of_targets(model, X, Y, device, batch_size=4096):
    out = []
    for i in range(0, len(Y), batch_size):
        x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
        lp = model.log2_probs(x)
        out.append(lp.gather(1, y[:, None]).squeeze(1).cpu())
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("text")
    p.add_argument("--emb_file", default=None, help="si le chemin a changé depuis l'entraînement")
    p.add_argument("--max_lines", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    model, vocab, cfg = load_model(args.model, device, args.emb_file)
    X, Y = prepare_data(args.text, vocab, cfg["k"], args.max_lines)

    lp = log2_probs_of_targets(model, X, Y, device)
    H = -lp.mean().item()
    print(f"entropie croisée : {H:.4f} bits/token")
    print(f"log2(perplexité) : {H:.4f}   (valeur demandée par le sujet, = entropie croisée)")
    print(f"perplexité       : {2 ** H:.2f}")

    # Les 2e et 3e </s> de chaque phrase sont triviaux à prédire et font baisser la
    # perplexité. Version sans eux, plus représentative du « vrai » texte :
    eos = vocab.dico_voca[EOS]
    trivial = (Y == eos) & (X[:, -1] == eos)
    H2 = -lp[~trivial].mean().item()
    print(f"perplexité sans les </s> de remplissage : {2 ** H2:.2f} "
          f"({int(trivial.sum())} cibles triviales retirées sur {len(Y)})")


if __name__ == "__main__":
    main()
