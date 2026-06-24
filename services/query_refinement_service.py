import json
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from typing import Optional

import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from spellchecker import SpellChecker


@dataclass
class RefinedQuery:
    original_query: str
    corrected_query: Optional[str] = None
    expanded_query: Optional[str] = None
    history_boosted_query: Optional[str] = None
    final_query: str = ""
    refinement_log: list[str] = field(default_factory=list)
    applied_refinements: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.final_query:
            self.final_query = self.history_boosted_query or self.expanded_query or self.corrected_query or self.original_query


class SearchHistory:
    def __init__(self, history_dir: Path, dataset: str):
        self.history_dir = history_dir
        self.dataset = dataset
        self.history_file = history_dir / f"search_history_{dataset}.json"
        self.history: list[dict] = []
        self._load()

    def _load(self):
        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.history = []
        else:
            self.history = []

    def record(self, query: str, top_doc_ids: list[str]):
        from datetime import datetime
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "top_doc_ids": top_doc_ids,
        }
        self.history.append(entry)
        self._save()

    def _save(self):
        self.history_dir.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_term_frequencies(self) -> Counter:
        stop_words = set(stopwords.words("english"))
        term_freq = Counter()

        for entry in self.history:
            query = entry.get("query", "").lower()
            tokens = word_tokenize(query)
            for token in tokens:
                if token.isalnum() and token not in stop_words and len(token) > 2:
                    term_freq[token] += 1

        return term_freq


class QueryRefinementService:
    def __init__(self, history_dir: Path):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        # Download NLTK resources if needed
        for resource in ["punkt", "wordnet", "stopwords", "averaged_perceptron_tagger"]:
            try:
                nltk.data.find(resource)
            except (LookupError, Exception):
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass  # Continue even if download fails

        self.spell_checker = SpellChecker()

    def spelling_correction(self, query: str) -> tuple[str, list[str]]:
        tokens = word_tokenize(query.lower())
        corrected_tokens = []
        corrections = []

        for token in tokens:
            if not token.isalnum():
                corrected_tokens.append(token)
                continue

            # Check if token is misspelled
            if len(token) < 3:
                corrected_tokens.append(token)
                continue

            corrected = self.spell_checker.correction(token)
            if corrected and corrected != token:
                corrected_tokens.append(corrected)
                corrections.append(f"{token} → {corrected}")
            else:
                corrected_tokens.append(token)

        corrected_query = " ".join(corrected_tokens)
        return corrected_query, corrections

    def synonym_expansion(self, query: str) -> tuple[str, list[str]]:
        tokens = word_tokenize(query.lower())
        pos_tags = pos_tag(tokens)
        original_tokens = set(tokens)
        added_synonyms = []

        for token, pos in pos_tags:
            if not token.isalnum() or len(token) < 3:
                continue

            # Map NLTK POS tags to WordNet POS tags
            wordnet_pos = None
            if pos.startswith("NN"):
                wordnet_pos = wordnet.NOUN
            elif pos.startswith("VB"):
                wordnet_pos = wordnet.VERB
            elif pos.startswith("JJ"):
                wordnet_pos = wordnet.ADJ
            elif pos.startswith("RB"):
                wordnet_pos = wordnet.ADV

            if wordnet_pos:
                synsets = wordnet.synsets(token, pos=wordnet_pos)
                synonym_count = 0
                for synset in synsets[:3]:  # Top 3 synsets
                    for lemma in synset.lemmas():
                        synonym = lemma.name().replace("_", " ")
                        if synonym != token and synonym not in original_tokens and synonym_count < 2:
                            added_synonyms.append(synonym)
                            synonym_count += 1
                            if synonym_count >= 2:
                                break
                    if synonym_count >= 2:
                        break

        # Limit total expansion
        added_synonyms = added_synonyms[:6]
        expanded_query = query + " " + " ".join(added_synonyms) if added_synonyms else query

        log = [f"Added {len(added_synonyms)} synonyms"] if added_synonyms else []
        return expanded_query.strip(), log

    def history_weighting(self, query: str, dataset: str) -> tuple[str, list[str]]:
        history = SearchHistory(self.history_dir, dataset)
        term_freq = history.get_term_frequencies()

        # Need at least 5 past queries for weighting
        if len(history.history) < 5 or not term_freq:
            return query, []

        # Calculate median term frequency
        if not term_freq.values():
            return query, []

        frequencies = sorted(term_freq.values())
        median_freq = frequencies[len(frequencies) // 2]

        # Boost underrepresented terms
        tokens = word_tokenize(query.lower())
        boosted_tokens = []

        for token in tokens:
            if token.isalnum() and len(token) > 2:
                freq = term_freq.get(token, 0)
                if freq > 0 and freq < median_freq:
                    boosted_tokens.append(token)

        boosted_query = query
        if boosted_tokens:
            boosted_query = query + " " + " ".join(boosted_tokens)

        log = [f"Boosted {len(boosted_tokens)} underrepresented terms"] if boosted_tokens else []
        return boosted_query.strip(), log

    def refine(
        self,
        query: str,
        dataset: str,
        enable_spelling_correction: bool = False,
        enable_synonym_expansion: bool = False,
        enable_search_history: bool = False,
    ) -> RefinedQuery:
        refined = RefinedQuery(original_query=query)

        # Stage 1: Spelling Correction
        if enable_spelling_correction:
            refined.corrected_query, corrections = self.spelling_correction(query)
            if corrections:
                refined.refinement_log.extend(corrections)
                refined.applied_refinements.append("spelling_correction")
            query = refined.corrected_query

        # Stage 2: Synonym Expansion
        if enable_synonym_expansion:
            refined.expanded_query, exp_log = self.synonym_expansion(query)
            if exp_log:
                refined.refinement_log.extend(exp_log)
                refined.applied_refinements.append("synonym_expansion")
            query = refined.expanded_query

        # Stage 3: History Weighting
        if enable_search_history:
            refined.history_boosted_query, hist_log = self.history_weighting(query, dataset)
            if hist_log:
                refined.refinement_log.extend(hist_log)
                refined.applied_refinements.append("search_history")
            query = refined.history_boosted_query

        refined.final_query = query
        return refined

    def record_search(self, query: str, dataset: str, top_doc_ids: list[str]):
        history = SearchHistory(self.history_dir, dataset)
        history.record(query, top_doc_ids)
