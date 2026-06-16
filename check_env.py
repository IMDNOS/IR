import nltk
import spacy
import torch
import ir_datasets
import sklearn
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("Python environment is working.")

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

nlp = spacy.load("en_core_web_sm")
print("spaCy model loaded.")

text = "Hello, this is the first IR project test."
print("NLTK tokens:", word_tokenize(text))
print("Stopwords sample:", stopwords.words("english")[:10])

print("All imports succeeded.")