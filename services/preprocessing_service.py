import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


@dataclass
class PreprocessingConfig:
    lowercase: bool = True
    remove_stopwords: bool = True
    use_lemmatization: bool = True
    use_stemming: bool = False
    keep_negations: bool = True


class PreprocessingService:
    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

        if self.config.keep_negations:
            self.stop_words -= {
                "no",
                "not",
                "nor",
                "never",
                "without",
                "against",
            }

    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def clean_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def remove_noise(self, text: str) -> str:
        text = re.sub(r"http\S+|www\S+", " ", text)
        return re.sub(r"\S+@\S+", " ", text)

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""

        text = self.normalize_unicode(str(text))
        text = self.remove_noise(text)
        text = self.clean_spaces(text)

        if self.config.lowercase:
            text = text.lower()

        return text

    def preprocess_for_embeddings(self, text: str) -> str:
        """
        Light preprocessing for BERT / SentenceTransformer models.
        Do not remove too much structure.
        """
        return self.normalize_text(text)

    def _tokens_from_spacy_doc(self, doc) -> list[str]:
        tokens = []

        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.like_url or token.like_email:
                continue

            token_text = token.text.lower().strip()
            if not token_text:
                continue
            if self.config.remove_stopwords and token_text in self.stop_words:
                continue

            if self.config.use_lemmatization:
                term = token.lemma_.lower().strip()
            else:
                term = token_text

            if self.config.use_stemming:
                term = self.stemmer.stem(term)

            if re.search(r"[a-zA-Z0-9]", term):
                tokens.append(term)

        return tokens

    def preprocess_for_lexical(self, text: str) -> list[str]:
        """
        Strong preprocessing for TF-IDF and BM25.
        Returns clean tokens.
        """
        normalized = self.normalize_text(text)
        return self._tokens_from_spacy_doc(self.nlp(normalized))

    def preprocess_for_tfidf(self, text: str) -> str:
        return " ".join(self.preprocess_for_lexical(text))

    def preprocess_document(self, doc_id: str, text: str) -> dict:
        lexical_tokens = self.preprocess_for_lexical(text)
        return self._document_payload(doc_id, text, lexical_tokens)

    def preprocess_query(self, query_id: str, text: str) -> dict:
        lexical_tokens = self.preprocess_for_lexical(text)
        return self._query_payload(query_id, text, lexical_tokens)

    def preprocess_documents_batch(
        self,
        documents: Iterable[tuple[str, str]],
        batch_size: int = 64,
    ) -> Iterable[dict]:
        items = list(documents)
        texts = [self.normalize_text(text) for _, text in items]

        for (doc_id, raw_text), spacy_doc in zip(
            items,
            self.nlp.pipe(texts, batch_size=batch_size),
        ):
            lexical_tokens = self._tokens_from_spacy_doc(spacy_doc)
            yield self._document_payload(doc_id, raw_text, lexical_tokens)

    def preprocess_queries_batch(
        self,
        queries: Iterable[tuple[str, str]],
        batch_size: int = 64,
    ) -> Iterable[dict]:
        items = list(queries)
        texts = [self.normalize_text(text) for _, text in items]

        for (query_id, raw_text), spacy_doc in zip(
            items,
            self.nlp.pipe(texts, batch_size=batch_size),
        ):
            lexical_tokens = self._tokens_from_spacy_doc(spacy_doc)
            yield self._query_payload(query_id, raw_text, lexical_tokens)

    def _document_payload(
        self,
        doc_id: str,
        raw_text: str,
        lexical_tokens: list[str],
    ) -> dict:
        return {
            "doc_id": str(doc_id),
            "raw_text": raw_text,
            "embedding_text": self.preprocess_for_embeddings(raw_text),
            "lexical_tokens": lexical_tokens,
            "lexical_text": " ".join(lexical_tokens),
        }

    def _query_payload(
        self,
        query_id: str,
        raw_text: str,
        lexical_tokens: list[str],
    ) -> dict:
        return {
            "query_id": str(query_id),
            "raw_text": raw_text,
            "embedding_text": self.preprocess_for_embeddings(raw_text),
            "lexical_tokens": lexical_tokens,
            "lexical_text": " ".join(lexical_tokens),
        }
