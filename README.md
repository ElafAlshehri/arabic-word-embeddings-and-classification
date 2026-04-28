# Arabic Word Embeddings and Text Classification

This project demonstrates a beginner Arabic NLP pipeline in the notebook:
- Train Arabic `Word2Vec` embeddings from Wikipedia text
- Build sentence vectors by averaging word embeddings
- Train a small PyTorch classifier on top of sentence vectors (toy labels for demo)

Notebook: `arabic_word_embeddings_text_classification.ipynb`

## Data Source

Arabic Wikipedia dump (official Wikimedia source):
- https://dumps.wikimedia.org/arwiki/latest/

The notebook uses the `arwiki-latest-pages-articles.xml.bz2` dump file.

## What the Notebook Does

1. Loads Arabic Wikipedia dump with `gensim.corpora.wikicorpus.WikiCorpus`
2. Preprocesses Arabic tokens by:
- normalizing Arabic letters (`أ/إ/آ -> ا`, `ى -> ي`, `ة -> ه`)
- removing diacritics (tashkeel)
- removing punctuation and non-Arabic characters
3. Trains `Word2Vec` (default shown):
- `sg=1` (Skip-gram)
- `vector_size=100`
- `window=5`
- `min_count=2`
- `epochs=10`
4. Compares a few Word2Vec setups (CBOW vs Skip-gram, 50 vs 100 dims)
5. Converts each sentence to one vector by averaging token vectors
6. Trains a simple feed-forward PyTorch classifier
7. Reports `accuracy`, `precision`, `recall`, and `F1`

## Important Note

Arabic Wikipedia is unlabeled for sentiment/classification. The notebook creates **synthetic toy labels** only to demonstrate classification code. For real classification, replace these labels with a real labeled dataset.

## Requirements

Install dependencies (example):

```bash
pip install gensim numpy scikit-learn torch
```

If you run in Google Colab, adapt file paths as needed (the current notebook path points to Google Drive).

## Run

Open and run the notebook:

```bash
jupyter notebook arabic_word_embeddings_text_classification.ipynb
```

or run it in Google Colab.
