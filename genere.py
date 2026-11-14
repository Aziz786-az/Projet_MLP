"""Génération de texte par tirage aléatoire dans les distributions du modèle.

    python genere.py mlp.pt --n 10 --temperature 1.0

Options :
    --n            nombre de phrases générées (default : 10)
    --max_len      longueur maximale d'une phrase en tokens (default : 60)
    --temperature  t < 1 plus prudent ; t > 1 plus varié (default : 1.0)
    --debut        début de phrase imposé, en tokens BPE séparés par des espaces
    --seed         en guise de reproductibilité des resultats
"""
import argparse

import torch

from common import BOS, EOS, detokenize, get_device, load_model, token_to_id


@torch.no_grad()
def generate(model, vocab, k, device, prefix=(), max_len=60, temperature=1.0):
    eos = vocab.dico_voca[EOS]
    context = [vocab.dico_voca[BOS]] * k + [token_to_id(vocab, t) for t in prefix]
    out = list(prefix)
    for _ in range(max_len):
        x = torch.tensor([context[-k:]], device=device)
        probs = torch.softmax(model(x)[0] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1).item()        # tirage aléatoire, pas argmax
        if nxt == eos:
            break
        out.append(vocab.word_array[nxt])
        context.append(nxt)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--n", type=int, default=10, help="nombre de phrases")
    p.add_argument("--max_len", type=int, default=60)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--debut", default="", help="début de phrase imposé (tokens BPE séparés par des espaces)")
    p.add_argument("--seed", type=int, default=None, help="graine du tirage : mêmes phrases à chaque lancement")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = get_device()
    model, vocab, cfg = load_model(args.model, device)
    for _ in range(args.n):
        toks = generate(model, vocab, cfg["k"], device, args.debut.split(), args.max_len, args.temperature)
        print(detokenize(toks))


if __name__ == "__main__":
    main()
