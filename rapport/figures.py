"""Figures du rapport, à partir des fichiers de resultats/.

fig_courbes.pdf : perte d'apprentissage et perplexité de validation par epoch (3 modèles)
fig_tsne.pdf    : projection t-SNE des 4 espaces d'embeddings (piste 2)
"""
import json
import math
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

OUT = "rapport/figures"
MODELES = {"mlp_base": "base (fournis, figés)",
           "mlp_emb_random": "piste 1 : appris, init. aléatoire",
           "mlp_emb_sgns": "piste 3 : appris, init. SGNS"}
COULEURS = {"mlp_base": "#1F4E79", "mlp_emb_random": "#C0504D", "mlp_emb_sgns": "#4F9A55"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def courbes():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for m, label in MODELES.items():
        h = json.load(open(f"resultats/{m}_history.json"))["history"]
        tr = [e for e in h if e["train"] is not None]
        ax1.plot([e["epoch"] for e in tr], [e["train"] / math.log(2) for e in tr],
                 marker="o", ms=3, color=COULEURS[m], label=label)
        va = [e for e in h if e["epoch"] >= 1]
        ax2.plot([e["epoch"] for e in va], [math.exp(e["val"]) for e in va],
                 marker="o", ms=3, color=COULEURS[m], label=label)
        best = min(va, key=lambda e: e["val"])
        ax2.plot(best["epoch"], math.exp(best["val"]), marker="*", ms=10, color=COULEURS[m])
    ax1.set(xlabel="epoch", ylabel="entropie croisée (bits/token)", title="Apprentissage")
    ax2.set(xlabel="epoch", ylabel="perplexité", title="Validation (étoile : modèle gardé)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.set_xticks(range(1, 6))
    ax1.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_courbes.pdf")


def charge_txt(path):
    with open(path, encoding="utf-8") as f:
        f.readline()
        mots, vecs = [], []
        for line in f:
            p = line.split()
            mots.append(p[0])
            vecs.append([float(x) for x in p[1:]])
    return mots, np.array(vecs)


def tsne():
    mots, fournis = charge_txt("donnees/bpe_embeddings_12k_32.txt")
    mots_s, sgns = charge_txt("resultats/sgns_32.txt")
    assert mots_s == mots
    espaces = {"embeddings fournis": fournis, "SGNS": sgns}
    for m, titre in (("mlp_emb_random", "MLP, init. aléatoire (piste 1)"),
                     ("mlp_emb_sgns", "MLP, init. SGNS (piste 3)")):
        espaces[titre] = torch.load(f"resultats/{m}.pt", map_location="cpu")["state"]["emb.weight"].numpy()

    groupes = {
        "couleurs": ["rouge", "jaune", "bleu", "vert", "noir", "blanc", "gris", "rose", "blanche", "noire"],
        "nombres": ["deux", "trois", "quatre", "cinq", "six", "sept", "huit", "dix", "douze", "cent"],
        "verbes de parole": ["dit", "répondit", "reprit", "murmura", "cria", "demanda", "ajouta", "répéta", "déclara"],
        "lieux": ["maison", "chambre", "cuisine", "boutique", "salle", "ville", "église", "jardin", "rue", "pièce"],
        "parenté": ["père", "mère", "fils", "fille", "frère", "soeur", "oncle", "tante", "mari", "femme"],
    }
    couleurs_g = ["#C0504D", "#1F4E79", "#4F9A55", "#8064A2", "#E36C09"]
    idx = {w: i for i, w in enumerate(mots)}
    freq = Counter(open("donnees/train_12k.bpe", encoding="utf-8").read().split())
    fond = [idx[w] for w, _ in freq.most_common(1500) if w in idx]
    cibles = {g: [idx[w] for w in ws if w in idx] for g, ws in groupes.items()}
    sel = sorted(set(fond) | {i for l in cibles.values() for i in l})
    pos = {i: j for j, i in enumerate(sel)}

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    for ax, (titre, E) in zip(axes.flat, espaces.items()):
        X = E[sel] / np.linalg.norm(E[sel], axis=1, keepdims=True)     # cosinus
        Y = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
        ax.scatter(Y[:, 0], Y[:, 1], s=2, color="#BBBBBB", lw=0)
        for (g, ids), c in zip(cibles.items(), couleurs_g):
            P = Y[[pos[i] for i in ids]]
            ax.scatter(P[:, 0], P[:, 1], s=14, color=c, label=g, lw=0)
            for i, (x, y) in zip(ids, P):
                ax.annotate(mots[i], (x, y), fontsize=5.5, color=c, xytext=(2, 1), textcoords="offset points")
        ax.set_title(titre)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("#CCCCCC")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower left", markerscale=1.2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_tsne.pdf")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    courbes()
    tsne()
    print("figures écrites dans", OUT)
