"""
Chapter 10: Natural Language Processing (NLP)
Section 10.2 — Techniques for Text Processing

Demonstrates core NLP text processing techniques:
- Tokenization
- Stop word removal
- Stemming and lemmatization
- TF-IDF vectorization
- Simple sentiment analysis with word polarity
"""

import re
from collections import Counter

# NLTK imports
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


# Sample corpus for demonstration
SAMPLE_TEXTS = [
    "Data science is transforming how businesses make critical decisions in today's competitive marketplace.",
    "Machine learning algorithms can process vast amounts of data to uncover hidden patterns and trends.",
    "Natural language processing enables computers to understand, interpret, and generate human language.",
    "Deep learning neural networks have achieved remarkable success in image recognition and speech processing.",
    "Effective data visualization communicates complex insights clearly to both technical and non-technical audiences.",
]


def tokenization_demo():
    """Demonstrate sentence and word tokenization."""
    print("=" * 60)
    print("STEP 1: Tokenization")
    print("=" * 60)

    text = " ".join(SAMPLE_TEXTS[:2])
    print(f"\nOriginal text:\n  \"{text}\"\n")

    # Sentence tokenization
    sentences = sent_tokenize(text)
    print(f"Sentence tokenization ({len(sentences)} sentences):")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")

    # Word tokenization
    tokens = word_tokenize(text.lower())
    print(f"\nWord tokenization ({len(tokens)} tokens):")
    print(f"  {tokens}")

    return tokens


def stopword_removal(tokens):
    """Remove stop words from tokenized text."""
    print("\n\n" + "=" * 60)
    print("STEP 2: Stop Word Removal")
    print("=" * 60)

    stop_words = set(stopwords.words("english"))
    print(f"\nTotal English stop words: {len(stop_words)}")
    print(f"Sample stop words: {sorted(list(stop_words))[:15]}...")

    # Filter: keep only alphabetic, non-stop-word tokens
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]

    removed = [w for w in tokens if w.isalpha() and w in stop_words]
    print(f"\nBefore: {len(tokens)} tokens")
    print(f"Removed: {len(removed)} stop words ({removed})")
    print(f"After:  {len(filtered)} tokens")
    print(f"  {filtered}")

    return filtered


def stemming_and_lemmatization(tokens):
    """Compare stemming vs lemmatization."""
    print("\n\n" + "=" * 60)
    print("STEP 3: Stemming vs Lemmatization")
    print("=" * 60)

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Demo words that highlight the difference
    demo_words = ["running", "businesses", "transforming", "decisions",
                  "processing", "algorithms", "hidden", "remarkable",
                  "effectively", "visualization"]

    print(f"\n  {'Original':<18} {'Stemmed':<18} {'Lemmatized':<18}")
    print("  " + "-" * 54)
    for word in demo_words:
        stemmed = stemmer.stem(word)
        lemmatized = lemmatizer.lemmatize(word)
        print(f"  {word:<18} {stemmed:<18} {lemmatized:<18}")

    # Apply to full token list
    stemmed_tokens = [stemmer.stem(w) for w in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in tokens]

    print(f"\n  Unique tokens (original):    {len(set(tokens))}")
    print(f"  Unique tokens (stemmed):     {len(set(stemmed_tokens))}")
    print(f"  Unique tokens (lemmatized):  {len(set(lemmatized_tokens))}")

    return lemmatized_tokens


def word_frequency_analysis(tokens):
    """Analyze word frequency distribution."""
    print("\n\n" + "=" * 60)
    print("STEP 4: Word Frequency Analysis")
    print("=" * 60)

    # Combine all texts and process
    full_text = " ".join(SAMPLE_TEXTS)
    all_tokens = word_tokenize(full_text.lower())
    stop_words = set(stopwords.words("english"))
    clean_tokens = [w for w in all_tokens if w.isalpha() and w not in stop_words]

    freq = Counter(clean_tokens)

    print(f"\nCorpus: {len(SAMPLE_TEXTS)} documents, {len(clean_tokens)} tokens")
    print(f"Unique words: {len(freq)}")
    print(f"\nTop 15 most frequent words:")
    for word, count in freq.most_common(15):
        bar = "█" * (count * 3)
        print(f"  {word:<18} {count:>3}  {bar}")


def tfidf_demo():
    """Demonstrate TF-IDF vectorization from scratch."""
    print("\n\n" + "=" * 60)
    print("STEP 5: TF-IDF Vectorization")
    print("=" * 60)

    import math

    stop_words = set(stopwords.words("english"))

    # Preprocess each document
    processed_docs = []
    for text in SAMPLE_TEXTS:
        tokens = word_tokenize(text.lower())
        filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
        processed_docs.append(filtered)

    # Build vocabulary
    vocab = sorted(set(word for doc in processed_docs for word in doc))
    n_docs = len(processed_docs)

    print(f"\n  Documents: {n_docs}")
    print(f"  Vocabulary size: {len(vocab)}")

    # Calculate TF-IDF for a sample of terms
    sample_terms = ["data", "learning", "language", "science", "processing"]

    print(f"\n  {'Term':<15} {'TF (Doc 1)':>10} {'DF':>5} {'IDF':>8} {'TF-IDF':>8}")
    print("  " + "-" * 50)

    for term in sample_terms:
        # Term frequency in document 1
        tf = processed_docs[0].count(term) / len(processed_docs[0]) if processed_docs[0] else 0
        # Document frequency
        df = sum(1 for doc in processed_docs if term in doc)
        # Inverse document frequency
        idf = math.log(n_docs / (df + 1)) + 1
        # TF-IDF
        tfidf = tf * idf

        print(f"  {term:<15} {tf:>10.4f} {df:>5} {idf:>8.4f} {tfidf:>8.4f}")

    print(f"\n  Interpretation: Higher TF-IDF = more distinctive to a document.")
    print(f"  Terms appearing in many documents get lower IDF scores.")


def simple_sentiment():
    """Demonstrate basic rule-based sentiment analysis."""
    print("\n\n" + "=" * 60)
    print("STEP 6: Simple Sentiment Analysis")
    print("=" * 60)

    positive_words = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "remarkable", "success", "effective", "powerful", "innovative",
        "transforming", "enables", "achieved", "clearly"
    }
    negative_words = {
        "bad", "poor", "terrible", "awful", "difficult", "complex",
        "challenging", "fail", "error", "problem", "risk", "bias"
    }

    reviews = [
        "Machine learning is a powerful and innovative approach to data analysis.",
        "The complex model produced poor results with terrible accuracy.",
        "Data visualization clearly and effectively communicates great insights.",
        "Bias in algorithms creates a difficult and challenging problem for fairness.",
        "The remarkable success of deep learning has been truly amazing.",
    ]

    print(f"\n  Analyzing {len(reviews)} sample texts:\n")

    for review in reviews:
        tokens = set(word_tokenize(review.lower()))
        pos_count = len(tokens & positive_words)
        neg_count = len(tokens & negative_words)
        score = pos_count - neg_count

        if score > 0:
            sentiment = "POSITIVE 😊"
        elif score < 0:
            sentiment = "NEGATIVE 😞"
        else:
            sentiment = "NEUTRAL 😐"

        print(f"  Text: \"{review[:70]}...\"")
        print(f"    Positive words: {pos_count}, Negative words: {neg_count}")
        print(f"    Sentiment: {sentiment} (score: {score:+d})\n")


if __name__ == "__main__":
    tokens = tokenization_demo()
    filtered = stopword_removal(tokens)
    lemmatized = stemming_and_lemmatization(filtered)
    word_frequency_analysis(lemmatized)
    tfidf_demo()
    simple_sentiment()
