# TP2 – Modèle de langage neuronal (MLP)

Abdelaziz MOHANDI, Mohamad AL CHARIF — M2 IAAA, Aix-Marseille Université

Modèle de langage (perceptron multicouche) qui prédit un token à partir des 3 tokens précédents,
avec les pistes 1, 2 et 3 du sujet. Le rapport complet est dans `rapport/Rapport_MLP.pdf`.

## Contenu du dossier

| Élément | Contenu |
|---|---|
| `Vocab.py` | classe fournie (inchangée) |
| `common.py` | préparation des données, chargement du vocabulaire et des modèles |
| `model.py` | le modèle (perceptron multicouche) |
| `train.py` | entraînement du modèle (modèle de base, pistes 1 et 3) |
| `perplexite.py` | perplexité d'un texte |
| `remplis_trous.py` | textes à trous (rang normalisé) |
| `genere.py` | génération de texte |
| `sgns.py` | plongements SGNS (pistes 2 et 3) |
| `similarite.py` | comparaison des plongements (piste 2) |
| `donnees/` | données utilisées (corpus, plongements fournis, textes à trous, triplets de similarité) |
| `resultats/` | modèles entraînés, logs et résultats des évaluations |
| `rapport/` | rapport (PDF), figures et script des figures (`figures.py`) |
| `colab/` | notebook Google Colab qui lance toutes les exécutions |

## Exécution sur Google Colab

1. Créer l'archive du code et des données, depuis ce dossier :
   ```bash
   zip -r Projet_MLP_colab.zip *.py donnees
   ```
2. Dans Google Drive, créer le dossier `MyDrive/Projet_MLP/` et y déposer `Projet_MLP_colab.zip`
   et `colab/Projet_MLP.ipynb`.
3. Ouvrir le notebook avec Colab, choisir un GPU (`Exécution > Modifier le type d'exécution > T4`),
   puis lancer `Exécution > Tout exécuter`.

Les résultats sont écrits dans `MyDrive/Projet_MLP/resultats/`.
