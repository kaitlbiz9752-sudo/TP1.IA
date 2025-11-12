## Pratique du NLP (Traitement du Langage Naturel)
## Objectifs pédagogiques

À la fin de ce TP, vous serez capable de :

Nettoyer et prétraiter un texte (tokenisation, stopwords, lemmatisation…)

Représenter des mots sous forme numérique (Bag of Words, TF-IDF)

Appliquer une analyse simple

Utiliser les bibliothèques : NLTK, spaCy, et scikit-learn

## Partie 1 — Préparation de l’environnement
**Objectif**

Installer et importer les bibliothèques nécessaires.

Étapes

Ouvrir un terminal dans VS Code.

Installer les bibliothèques :

```bash
pip install pandas nltk spacy scikit-learn
python -m spacy download en_core_web_sm
```
---

Importer les modules et télécharger les ressources NLTK :

```python
import nltk, spacy
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nlp = spacy.load("en_core_web_sm")
```

## Questions

Quelle est la commande pour installer une bibliothèque avec pip ?
→ pip install nom_de_la_bibliotheque

Quelle est la différence entre NLTK et spaCy ?
→ NLTK : bibliothèque modulaire et académique.
→ spaCy : pipeline industriel complet et rapide.

---

## Partie 2 — Nettoyage du texte
**Objectif**

Normaliser un texte en supprimant la ponctuation et en le mettant en minuscules.

Étapes

Créer une variable text avec des majuscules, des ponctuations, etc.

Convertir tout le texte en minuscules.

Supprimer la ponctuation avec regex ou une méthode string.

Appliquer le même nettoyage à la colonne texte du dataset.

**CODE**

```python
text = "Hello World! NLP is AMAZING, isn't it?"
text_clean = text.lower()
text_clean = re.sub(r'[^\w\s]', '', text_clean)
```
**AFFICHAGE**



<img width="653" height="72" alt="image" src="https://github.com/user-attachments/assets/3dd5a7b0-fb7b-4880-ab48-ebafa05721b4" />

## Questions
Pourquoi mettre le texte en minuscules ?
→ Pour éviter les doublons entre “ChatGPT” et “chatgpt”.

Exemple avant/après nettoyage :

```bash
Avant : Hello World! NLP is AMAZING.
Après  : hello world nlp is amazing
```

---


## Partie 3 — Tokenisation et Stopwords
**Objectif**

Découper le texte en mots et éliminer les mots vides.

**Étapes**

Tokeniser le texte : transformer les phrases en liste de mots.
Utiliser wordpunct_tokenize (plus stable que word_tokenize) ou spaCy.

Charger la liste des stopwords de NLTK.

Retirer ces mots du texte.

(Optionnel) Ajouter vos propres stopwords personnalisés.

**CODE :**

```python
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
stop_en = set(stopwords.words('english'))

df["tokens"] = df["clean_text"].apply(wordpunct_tokenize)
df["tokens_no_stop"] = df["tokens"].apply(lambda t: [w for w in t if w.lower() not in stop_en])
```
**AFFICHAGE**




<img width="654" height="440" alt="image" src="https://github.com/user-attachments/assets/f33fa4d5-a2e8-4a53-bde0-2bf91947ae33" />


## Questions

Que représentent les stopwords ?
→ Des mots très fréquents à faible valeur sémantique (ex : the, and, is…)

Quelle proportion du texte est supprimée ?
→ En moyenne, 30 à 40 % pour l’anglais.

---


## Partie 4 — Lemmatisation
**Objectif**

Réduire les mots à leur forme de base (ex : running → run).

**Étapes**

Charger le modèle linguistique spaCy :

```python
import spacy
nlp = spacy.load("en_core_web_sm")
```

Définir une fonction pour appliquer la lemmatisation :

```python
def lemmatize_tokens(tokens):
    doc = nlp(" ".join(tokens))
    return [tok.lemma_ for tok in doc]
```


Appliquer cette fonction à vos tokens filtrés.

**CODE :**

```python
df["lemmas"] = df["tokens_no_stop"].apply(lemmatize_tokens)
```

**AFFICHAGE**




<img width="565" height="398" alt="image" src="https://github.com/user-attachments/assets/290d1129-a2c5-4f98-8bc3-60b9df229001" />


**Questions**

Différence entre stemming et lemmatisation ?
→ Stemming coupe les suffixes sans contexte, lemmatisation analyse la grammaire.

Pourquoi est-ce important ?
→ Pour regrouper les formes d’un même mot et améliorer la compréhension sémantique.

---


## Partie 5 — Représentations numériques (optionnelle)
**Objectif**

Convertir les textes en vecteurs pour un modèle de machine learning.

**Étapes**


```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(df["clean_text"])

vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(df["clean_text"])
```

**AFFICHAGE**




<img width="523" height="69" alt="image" src="https://github.com/user-attachments/assets/557ef153-0138-4dd4-b927-ad7aae5a267f" />

---


## Partie 6 — Analyse simple (optionnelle)

**Exemple : classification spam/ham**

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, df["label"], test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```
**AFFICHAGE**




<img width="572" height="249" alt="image" src="https://github.com/user-attachments/assets/26d048d9-875d-4cc7-bdb5-53331918b244" />

---

## Résumé du pipeline NLP

| Étape | Action | Bibliothèque |
|:------|:--------|:--------------|
| **1** | Importer et installer les outils | NLTK, spaCy, scikit-learn |
| **2** | Nettoyer le texte | re / pandas |
| **3** | Tokeniser et retirer les stopwords | NLTK |
| **4** | Lemmatiser les mots | spaCy |
| **5** | Transformer en vecteurs | scikit-learn |
| **6** | Analyser ou classifier | scikit-learn |

---

## Voici un vidéo déscriptive de mon travaille :

https://github.com/user-attachments/assets/89f199e1-d69c-4828-b3fa-8d889e37b3ae


