"""
Mini-replication of 'Identify and Neutralize' algorithm from Man is to Computer
Programmer as Woman is to Homemaker? Debiasing Word Embeddings (Bolukbasi et al., 2016)
"""
# Imports
# ------------------------------------------------------------------------------
import torch as t
import numpy as np
import einops
import gensim.downloader
from gensim.models import KeyedVectors
# ------------------------------------------------------------------------------

# Load and unit normalize google news word2vec embeddings
# ------------------------------------------------------------------------------
word_vectors = gensim.downloader.load("word2vec-google-news-300")
words = list(word_vectors.index_to_key[:3000000])
vectors = t.tensor(np.array([word_vectors[word] for word in words]), dtype=t.float32)
vectors_normalized = vectors / t.norm(vectors, dim=1, keepdim=True)
new_vectors = KeyedVectors(300)
new_vectors.add_vectors(words, vectors_normalized.numpy())
# ------------------------------------------------------------------------------

# man:woman::computer_programmer:?
print(f"Biased word embeddings: Man is to woman as computer programmer is to {max(word_vectors.most_similar(positive=['woman', 'computer_programmer'], negative=['man']), key=lambda x: x[1])[0]}")

# Identify gender subspace
# ------------------------------------------------------------------------------
pairs = [
    ("she", "he"),
    ("daughter", "son"),
    ("her", "his"),
    ("mother", "father"),
    ("woman", "man"),
    ("gal", "guy"),
    ("mary", "john"),
    ("girl", "boy"),
    ("herself", "himself"),
    ("female", "male"),
]

diff_vectors = []
for fem, masc in pairs:
    fem_vec = t.tensor(new_vectors[fem])
    masc_vec = t.tensor(new_vectors[masc])
    diff = fem_vec - masc_vec
    diff_vectors.append(diff)

diff_matrix = t.stack(diff_vectors)
U, S, V = t.svd(diff_matrix)
gender_direction = V[:, 0]  # First right singular vector is first principal component
gender_direction = gender_direction / t.norm(gender_direction)
# ------------------------------------------------------------------------------

# Project embeddings onto orthogonal subspace of gender direction
# -----------------------------------------------------------------------------
def neutralize(vectors, gender_direction):
    projs = einops.einsum(vectors, gender_direction.unsqueeze(1), 'vocab d, d uno -> vocab uno') * gender_direction.unsqueeze(0)
    neutralized = vectors - projs
    neutralized = neutralized / t.norm(neutralized, dim=1, keepdim=True)
    return neutralized

vectors_debiased = neutralize(vectors_normalized, gender_direction)
debiased_vectors = KeyedVectors(300)
debiased_vectors.add_vectors(words, vectors_debiased.numpy())
# ------------------------------------------------------------------------------

# Sanity check
# -----------------------------------------------------------------------------
check_dots = einops.einsum(vectors_debiased, gender_direction, 'v d, d -> v').abs()
assert check_dots.max() < 1e-4
# ------------------------------------------------------------------------------

# man:woman::computer_programmer:?
# ------------------------------------------------------------------------------
print(f"Debiased word embeddings: Man is to woman as computer programmer is to {max(debiased_vectors.most_similar(positive=['woman', 'computer_programmer'], negative=['man']), key = lambda x: x[1])[0]}")
# ------------------------------------------------------------------------------

