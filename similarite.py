"""Piste 2 : comparaison de plusieurs espaces d'embeddings.

    python similarite.py --emb resultats/sgns_32.txt resultats/mlp_emb_random.pt bpe_embeddings_12k_32.txt \
        --sim Le_comte_de_Monte_Cristo.100.sim --corpus train_12k.bpe
"""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F

from common import BOS, EOS, load_vocab

SPECIAL = {BOS, EOS, "<unk>"}
VOISINS = 10                                           
CIBLES = 1000                                          
EXEMPLES = ["il", "maison", "rouge", "dit", "deux", "paris"]  


def load_source(path):
    """Renvoie (nom, liste des tokens, dico token -> indice, matrice (|V|, d))."""
    if path.endswith(".pt"):
        ckpt = torch.load(path, map_location="cpu")
        vocab = load_vocab(ckpt["config"]["emb_file"])
        E = ckpt["state"]["emb.weight"].float()
    else:
        vocab = load_vocab(path)
        E = vocab.matrice.float()
    return Path(path).stem, vocab.word_array, vocab.dico_voca, E


def corpus_stats(path):
    """Compte les tokens et, pour chaque mot recollé, ses découpages BPE observés."""
    tok_counts = Counter()
    segs = defaultdict(Counter)
    with open(path, encoding="utf-8") as f:
        for line in f:
            toks = line.split()
            tok_counts.update(toks)
            word = []
            for t in toks + [BOS]:                      
                if t.startswith("#") and len(t) > 1 and word:
                    word.append(t)
                    continue
                if word:
                    segs["".join(w.lstrip("#") for w in word)][tuple(word)] += 1
                word = [t]
    return tok_counts, segs


def segment(word, segs):
    """Découpage le plus fréquent du mot dans le corpus, ou None s'il en est absent."""
    for form in (word, word.replace("œ", "oe").replace("æ", "ae")):
        if form in segs:
            return list(segs[form].most_common(1)[0][0])
    return None


def load_triplets(path):
    with open(path, encoding="utf-8") as f:
        return [tuple(line.split()) for line in f if len(line.split()) == 3]


def word_vector(pieces, dico, E):
    """Moyenne des vecteurs des sous-tokens du mot (un seul morceau = son vecteur)."""
    if pieces is None or any(p not in dico for p in pieces):
        return None
    return E[[dico[p] for p in pieces]].mean(0)


def similarity_scores(triplets, seg_of, dico, E):
    passed = covered = 0
    for triplet in triplets:
        vecs = [word_vector(seg_of[w], dico, E) for w in triplet]
        if any(v is None for v in vecs):
            continue
        covered += 1
        m, mp, mm = vecs
        if F.cosine_similarity(m, mp, dim=0) > F.cosine_similarity(m, mm, dim=0):
            passed += 1
    return passed, covered


def neighbours(E, idx, n):
    """Indices des n plus proches voisins (cosinus) de chaque ligne idx."""
    En = F.normalize(E, dim=1)
    sims = En[idx] @ En.T
    sims[torch.arange(len(idx)), idx] = -2.0
    return sims.topk(n, dim=1).indices


def neighbour_overlap(src_a, src_b, targets, n):
    """Proportion moyenne des n plus proches voisins communs aux deux espaces."""
    _, words_a, dico_a, E_a = src_a
    _, _, dico_b, E_b = src_b
    common = [w for w in words_a if w in dico_b]           
    Ea = E_a[[dico_a[w] for w in common]]
    Eb = E_b[[dico_b[w] for w in common]]
    pos = {w: i for i, w in enumerate(common)}
    idx = torch.tensor([pos[t] for t in targets if t in pos])
    na, nb = neighbours(Ea, idx, n), neighbours(Eb, idx, n)
    overlaps = [len(set(a.tolist()) & set(b.tolist())) / n for a, b in zip(na, nb)]
    return sum(overlaps) / len(overlaps), len(overlaps)


def show_neighbours(sources, words, n):
    for w in words:
        print(f"  {w}")
        for name, word_array, dico, E in sources:
            if w not in dico:
                continue
            best = neighbours(E, torch.tensor([dico[w]]), n)[0]
            print(f"    {name:>28} : " + " ".join(word_array[j] for j in best))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb", nargs="+", required=True, help="fichiers .txt ou modèles .pt")
    p.add_argument("--sim", nargs="+", required=True, help="fichiers de triplets du TP1")
    p.add_argument("--corpus", required=True, help="corpus BPE (découpage des mots, fréquences)")
    args = p.parse_args()

    sources = [load_source(path) for path in args.emb]
    tok_counts, segs = corpus_stats(args.corpus)

    for sim_path in args.sim:
        triplets = load_triplets(sim_path)
        words = sorted({w for t in triplets for w in t})
        seg_of = {w: segment(w, segs) for w in words}
        split = [w for w in words if seg_of[w] is not None and seg_of[w] != [w]]
        unseen = [w for w in words if seg_of[w] is None]
        print(f"\n=== {sim_path} : {len(triplets)} triplets, {len(words)} mots")
        if split:
            print("  découpage : " + ", ".join(f"{w} = {' '.join(seg_of[w])}" for w in split))
        if unseen:
            print("  absents du corpus (triplets ignorés) : " + ", ".join(unseen))
        print(f"  {'source':>28} | réussis/couverts")
        for name, _, dico, E in sources:
            ok, cov = similarity_scores(triplets, seg_of, dico, E)
            print(f"  {name:>28} | {ok:3d}/{cov:3d} = {ok / cov if cov else 0:.3f}")

    if len(sources) >= 2:
        targets = [t for t, _ in tok_counts.most_common() if t not in SPECIAL][:CIBLES]
        print(f"\n=== voisins communs ({VOISINS} plus proches, {len(targets)} tokens les plus fréquents)")
        for a, b in combinations(sources, 2):
            ov, n = neighbour_overlap(a, b, targets, VOISINS)
            print(f"  {a[0]:>28} / {b[0]:<28} : {ov:.3f}  ({n} tokens)")
    print(f"\n=== {VOISINS} plus proches voisins")
    show_neighbours(sources, EXEMPLES, VOISINS)


if __name__ == "__main__":
    main()
