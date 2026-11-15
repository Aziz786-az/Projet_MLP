"""Fonctions partagées : vocabulaire, lecture des corpus, chargement des modèles."""
import os
import torch

from Vocab import Vocab

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def get_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vocab(emb_file):
    """Charge le fichier d'embeddings avec la classe Vocab fournie.
    vocab.dico_voca : token -> indice ; vocab.matrice : tenseur (|V|, d)."""
    return Vocab(emb_filename=emb_file)


def token_to_id(vocab, tok):
    """Indice d'un token ; <unk> s'il est inconnu (None si <unk> n'existe pas)."""
    idx = vocab.dico_voca.get(tok)
    if idx is None:
        idx = vocab.dico_voca.get(UNK)
    return idx


def sentence_tokens(line, k):
    """Normalise une ligne du corpus : k <s> au début, k </s> à la fin.
    Pour k = 3, c'est exactement le format des fichiers fournis."""
    toks = line.split()
    while toks and toks[0] == BOS:
        toks.pop(0)
    while toks and toks[-1] == EOS:
        toks.pop()
    if not toks:
        return None
    return [BOS] * k + toks + [EOS] * k


def prepare_data(path, vocab, k, max_lines=None, verbose=True):
    """Étape 1 du TP : transforme un texte en exemples de k+1 indices consécutifs.

    Renvoie X (N, k) = les k tokens de contexte, Y (N,) = le token à prédire.
    Les fenêtres ne chevauchent jamais deux phrases (une phrase = une ligne).
    """
    ids, sent = [], []
    n_lines = n_unk_lines = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_lines is not None and n_lines >= max_lines:
                break
            toks = sentence_tokens(line, k)
            if toks is None:
                continue
            line_ids = [token_to_id(vocab, t) for t in toks]
            if any(i is None for i in line_ids):   # token inconnu et pas de <unk> : on saute la phrase
                n_unk_lines += 1
                continue
            ids.extend(line_ids)
            sent.extend([n_lines] * len(line_ids))
            n_lines += 1

    ids = torch.tensor(ids, dtype=torch.long)
    sent = torch.tensor(sent, dtype=torch.long)
    windows = ids.unfold(0, k + 1, 1)            # (T-k, k+1) : toutes les fenêtres
    sent_w = sent.unfold(0, k + 1, 1)
    keep = sent_w[:, 0] == sent_w[:, -1]         # on garde celles qui restent dans une phrase
    windows = windows[keep]
    if verbose:
        print(f"{path} : {n_lines} phrases, {len(ids)} tokens, {len(windows)} exemples"
              + (f", {n_unk_lines} phrases ignorées (token inconnu)" if n_unk_lines else ""))
    return windows[:, :k].contiguous(), windows[:, k].contiguous()


def cached_data(path, vocab, k, max_lines=None):
    """prepare_data avec cache disque (le prétraitement du train prend du temps)."""
    suffix = f".k{k}" + (f".n{max_lines}" if max_lines else "") + ".pt"
    cache = path + suffix
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        X, Y = torch.load(cache)
        print(f"{path} : {len(Y)} exemples (cache)")
        return X, Y
    X, Y = prepare_data(path, vocab, k, max_lines)
    try:
        torch.save((X, Y), cache)
    except OSError:
        pass
    return X, Y


def load_model(ckpt_path, device, emb_file=None):
    """Recharge un modèle sauvegardé par train.py, avec son vocabulaire."""
    from model import MLPLangModel
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    vocab = load_vocab(emb_file or cfg["emb_file"])
    model = MLPLangModel(cfg["vocab_size"], cfg["emb_dim"], cfg["k"], cfg["dh"])
    model.load_state_dict(ckpt["state"])
    model.to(device).eval()
    return model, vocab, cfg


def detokenize(tokens):
    """Recolle les sous-mots BPE : un token qui commence par # se colle au précédent."""
    out = ""
    for t in tokens:
        if t.startswith("#") and len(t) > 1:
            out += t[1:]
        else:
            out += (" " if out else "") + t
    return out
