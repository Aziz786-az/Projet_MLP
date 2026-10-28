import os
import json
import torch
import numpy as np
import sys 

class Vocab:
    def __init__(self, **kwargs):
        self.dico_voca = {}
        self.word_array = []

        # --- NOUVELLE STRATÉGIE : Initialisation via un fichier de vocabulaire global BPE ---
        if "vocab_filename" in kwargs and kwargs["vocab_filename"] not in [None, "", "null"]:
            fichier_vocab = kwargs["vocab_filename"]
            self.emb_dim = kwargs["emb_dim"]
            nb_tokens = 0
            
            # Si le fichier fourni est le .json d'origine, on le convertit d'abord en .txt
            if fichier_vocab.endswith('.json'):
                fichier_vocab_txt = fichier_vocab.replace('.json', '.txt')
                if not os.path.exists(fichier_vocab_txt):
                    self._convert_json_to_txt(fichier_vocab, fichier_vocab_txt)
                fichier_vocab = fichier_vocab_txt # On bascule sur le fichier texte pour la lecture
            
            print(fichier_vocab)
            # Lecture du fichier texte indexé
            with open(fichier_vocab, 'r', encoding='utf-8') as fi:
                for line in fi:
                    line = line.strip()
                    if not line:
                        continue
                    
                    
                    # Gestion du format "index mot" ou juste "mot"
                    tokens = line.split()
                    token = tokens[1] if len(tokens) > 1 else tokens[0]
                    
                    if token not in self.dico_voca:
                        self.word_array.append(token)
                        self.dico_voca[token] = nb_tokens
                        nb_tokens += 1
            
            # Sécurité : On s'assure que <unk> est présent pour get_word_index2
            if '<unk>' not in self.dico_voca:
                self.word_array.append('<unk>')
                self.dico_voca['<unk>'] = nb_tokens
                nb_tokens += 1

            self.vocab_size = nb_tokens
            print("vocab size (depuis fichier global) =", self.vocab_size, "emb_dim =", self.emb_dim)
            self.matrice = torch.zeros((self.vocab_size, self.emb_dim))

        # --- Structures existantes  ---
        elif "list_filename" in kwargs:
            with open(kwargs["list_filename"], 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = line.strip().split(" ")
                    self.dico_voca[tokens[1]] = len(self.word_array)
                    self.word_array.append(tokens[1])
        elif "emb_filename" in kwargs and kwargs["emb_filename"] not in [None, "", "null"]:
            with open(kwargs["emb_filename"], 'r', encoding='utf-8') as fi:
                ligne = fi.readline().strip()
                (self.vocab_size, self.emb_dim) = map(int, ligne.split(" "))
                print(self.vocab_size, self.emb_dim)

                self.matrice = torch.zeros((self.vocab_size, self.emb_dim))
                indice_mot = 0
        
                ligne = fi.readline().strip()

                while ligne != '': 
                    splitted_ligne = ligne.split()
                    self.dico_voca[splitted_ligne[0]] = indice_mot
                    self.word_array.append(splitted_ligne[0])
                    for i in range(1, len(splitted_ligne)):
                        self.matrice[indice_mot, i-1] = float(splitted_ligne[i])
                    indice_mot += 1
                    ligne = fi.readline().strip()
                    
        else:
            # Mode Scratch par défaut (secours sur le fichier train)
            fichier_corpus = kwargs["corpus_filename"]
            self.emb_dim = kwargs["emb_dim"]
            nb_tokens = 0
            with open(fichier_corpus, 'r', encoding='utf-8') as fi:
                for line in fi:
                    line = line.rstrip()
                    tokens = line.split(" ")
                    for token in tokens:
                        if token not in self.dico_voca:
                            self.word_array.append(token)
                            self.dico_voca[token] = nb_tokens
                            nb_tokens += 1
            
            if '<unk>' not in self.dico_voca:
                self.word_array.append('<unk>')
                self.dico_voca['<unk>'] = nb_tokens
                nb_tokens += 1
                
            self.vocab_size = nb_tokens
            print("vocab size (depuis corpus) =", self.vocab_size, "emb_dim =", self.emb_dim)
            self.matrice = torch.zeros((self.vocab_size, self.emb_dim))

    def _convert_json_to_txt(self, json_path, txt_path):
        """ Fonction interne pour extraire et nettoyer le JSON BPE HuggingFace """
        print(f"Extraction et conversion automatique de {json_path} vers {txt_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab_dict = data["model"]["vocab"]
        # Tri par ID pour garder l'ordre exact du tokenizer
        sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
        
        with open(txt_path, 'w', encoding='utf-8') as f_out:
            for token, token_id in sorted_vocab:
                # Harmonisation du token inconnu pour correspondre à get_word_index2
                if token == "[UNK]":
                    token = "<unk>"
                f_out.write(f"{token_id} {token}\n")

    def get_word_index(self, mot):
        if not mot in self.dico_voca:
            print(f"ValueError : mot {mot} introuvable")
            return None
        return self.dico_voca[mot]
                
    def get_word_index2(self, mot):
        #print(mot)
        if not mot in self.dico_voca:
            sys.stderr.write("\n")
            sys.stderr.write(mot)
            return self.dico_voca['<unk>']
        return self.dico_voca[mot]
                
    def get_emb(self, mot):
        if not mot in self.dico_voca:
            return None
        return self.matrice[self.dico_voca[mot]]
    
    def get_emb_torch(self, indice_mot):
        return self.matrice[indice_mot]
        
    def get_one_hot(self, mot):
        vect = torch.zeros(len(self.dico_voca))
        vect[self.dico_voca[mot]] = 1
        return vect

    def get_word(self, index):
        if index < len(self.word_array):
            return self.word_array[index]
        return None

    def save_as_list(self, vocab_filename):
        with open(vocab_filename, "w", encoding='utf-8') as f:
            for i in range(len(self.word_array)):
                print(i, self.word_array[i], file=f)

