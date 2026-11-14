"""Texte à trous : retrouver le token masqué parmi 100 candidats, mesuré par le rang normalisé.

    python remplis_trous.py mlp.pt file_1.mask tokens_candidats_textes_a_trou_12k/
    python remplis_trous.py mlp.pt file_1.mask tokens_candidats_textes_a_trou_12k/<fichier> --contexte gauche
    
"""
import argparse
import os

import torch

from common import BOS, EOS, get_device, load_model, token_to_id


def read_candidates(path):
    with open(path, encoding="utf-8") as f:
        return [line.split()[-1] for line in f if line.strip()]


def read_masks(path):
    """Chaque ligne non vide : position du trou puis la séquence de tokens."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.replace("}", " ").split()
            if len(parts) < 2:
                continue
            examples.append((int(parts[0]), parts[1:]))
    return examples


def fix_candidates(cands, answer, n=100):
    """Le bon token doit être dans la liste ; sinon on le rajoute à la place d'un autre."""
    cands = list(dict.fromkeys(cands))[:n]      # sans doublons, ordre conservé
    if answer in cands:
        return cands
    return [c for c in cands if c != answer][:n - 1] + [answer]


@torch.no_grad()
def score_candidates(model, vocab, tokens, hole, cands, k, device, contexte="complet"):
    ids = [token_to_id(vocab, t) for t in tokens]
    cand_ids = [token_to_id(vocab, c) for c in cands]
    last = hole if contexte == "gauche" else min(hole + k, len(ids) - 1)
    targets = range(hole, last + 1)             
    ctxs, tgts = [], []
    for c in cand_ids:
        seq = ids.copy()
        seq[hole] = c
        for j in targets:
            ctxs.append(seq[j - k:j])
            tgts.append(seq[j])
    x = torch.tensor(ctxs, device=device)
    y = torch.tensor(tgts, device=device)
    lp = model.log2_probs(x).gather(1, y[:, None]).squeeze(1)
    return lp.view(len(cand_ids), len(targets)).sum(1).cpu()


def normalized_rank(scores, correct_idx):
    s = scores[correct_idx]
    rank = int((scores > s).sum()) + 1          
    return rank, 1 - (rank - 1) / (len(scores) - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("masks", nargs="+", help="fichiers .mask")
    p.add_argument("--candidats", required=True, help="un fichier de candidats ou le répertoire des 10 fichiers")
    p.add_argument("--contexte", choices=["complet", "gauche"], default="complet")
    p.add_argument("--top", type=int, default=5, help="afficher les meilleurs candidats")
    p.add_argument("--base", type=int, default=0, choices=[0, 1],
                   help="la position du trou est comptée à partir de 0 (défaut, fichiers fournis) ou de 1 (exemple du sujet)")
    p.add_argument("--emb_file", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    model, vocab, cfg = load_model(args.model, device, args.emb_file)
    k = cfg["k"]

    if os.path.isdir(args.candidats):
        cand_files = sorted(os.path.join(args.candidats, f) for f in os.listdir(args.candidats))
    else:
        cand_files = [args.candidats]

    all_ranks = []
    for mask_file in args.masks:
        for pos, tokens in read_masks(mask_file):
            hole = pos - args.base              
            if not 0 <= hole < len(tokens):
                print(f"\n{mask_file} : position {pos} hors de la ligne ({len(tokens)} tokens), ignorée")
                continue
            answer = tokens[hole]
            # si la ligne n'a pas assez de contexte, on complète avec <s> / </s>
            n_left = max(0, k - hole)
            tokens = [BOS] * n_left + tokens + [EOS] * k
            hole += n_left
            ctx = " ".join(tokens[max(0, hole - 6):hole]) + " [" + answer + "] " + \
                " ".join(tokens[hole + 1:hole + 7])
            print(f"\n{mask_file} : trou n°{pos} -> réponse « {answer} »   ... {ctx} ...")
            for cf in cand_files:
                raw = read_candidates(cf)
                added = answer not in raw
                cands = fix_candidates(raw, answer)
                scores = score_candidates(model, vocab, tokens, hole, cands, k, device, args.contexte)
                rank, nr = normalized_rank(scores, cands.index(answer))
                all_ranks.append(nr)
                best = [cands[i] for i in scores.argsort(descending=True)[:args.top]]
                print(f"  {os.path.basename(cf)} : rang {rank}/{len(cands)}  "
                      f"rang normalisé {nr:.3f}{' (réponse ajoutée)' if added else ''}  | meilleurs : {' '.join(best)}")

    print(f"\nrang normalisé moyen ({args.contexte}) : {sum(all_ranks) / len(all_ranks):.3f} "
          f"sur {len(all_ranks)} évaluations")


if __name__ == "__main__":
    main()
