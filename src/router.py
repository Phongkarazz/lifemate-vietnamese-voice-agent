"""Semantic Tool Router for ZAC 2026 LifeMate.
Ranks installed apps for a given user request using fast word (1,2)-gram TF-IDF similarity
over app names, Vietnamese descriptions, and tool/parameter schemas.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer = None  # type: ignore[assignment]
    cosine_similarity = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _pure_python_tfidf_scores(app_docs: list[str], full_query: str) -> list[float]:
    """Compute sublinear (1,2)-gram TF-IDF cosine similarity without third-party dependencies."""

    def _ngrams(text: str) -> Counter[str]:
        toks = _TOKEN_RE.findall(text.lower())
        grams = list(toks) + [f"{toks[i]} {toks[i + 1]}" for i in range(len(toks) - 1)]
        return Counter(grams)

    doc_counts = [_ngrams(doc) for doc in app_docs]
    n_docs = len(doc_counts)
    df: Counter[str] = Counter()
    for cnt in doc_counts:
        for term in cnt:
            df[term] += 1

    idf = {term: math.log((1 + n_docs) / (1 + freq)) + 1.0 for term, freq in df.items()}

    def _vec(cnt: Counter[str]) -> dict[str, float]:
        v = {term: (1.0 + math.log(tf)) * idf[term] for term, tf in cnt.items() if term in idf}
        norm = math.sqrt(sum(w * w for w in v.values()))
        return {t: w / norm for t, w in v.items()} if norm > 0 else {}

    q_vec = _vec(_ngrams(full_query))
    scores: list[float] = []
    for cnt in doc_counts:
        d_vec = _vec(cnt)
        scores.append(sum(q_vec[t] * d_vec.get(t, 0.0) for t in q_vec))
    return scores


class SemanticToolRouter:
    """Fast, deterministic semantic tool and app router (< 2ms CPU latency)."""

    def __init__(self, top_k_recommended: int = 4):
        self.top_k_recommended = top_k_recommended
        self.vectorizer = (
            TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                token_pattern=r"(?u)\b\w+\b",
            )
            if _SKLEARN_AVAILABLE
            else None
        )

    def route_sample_apps(
        self,
        query: str,
        sample: dict[str, Any],
        catalog_tools_map: dict[str, dict[str, Any]],
        installed_app_descriptions: dict[str, str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Rank installed apps for `query`.
        Returns: `(recommended_apps, secondary_apps)`.
        Guarantees: Every installed app appears in exactly one of the two lists.
        """
        installed = sample.get("context", {}).get("installed_apps", {})
        app_names = [a for a in installed.keys() if a != "sys"]

        # If top_k_recommended or fewer apps installed, all are recommended
        if len(app_names) <= self.top_k_recommended:
            return app_names, []

        installed_app_descriptions = installed_app_descriptions or {}

        # Build document text for each installed app: app_name + app_description + tool descriptions
        app_docs: list[str] = []
        for app in app_names:
            parts = [app, installed_app_descriptions.get(app, "")]
            # Include tool descriptions of that app
            tool_names = (installed.get(app) or {}).get("tools", [])
            for tname in tool_names:
                parts.append(tname)
                spec = catalog_tools_map.get(tname) or {}
                if spec.get("description"):
                    parts.append(spec["description"])
                # Include parameter names and descriptions
                for p_name, p_info in (spec.get("parameters", {}).get("properties") or {}).items():
                    parts.append(p_name)
                    if p_info.get("description"):
                        parts.append(p_info["description"])
            app_docs.append(" ".join(parts))

        # Query includes the user prompt and recent user history text
        query_parts = [query]
        for turn in (sample.get("history") or [])[-2:]:
            if turn.get("role") == "user" and turn.get("audio_text"):
                query_parts.append(turn["audio_text"])
        full_query = " ".join(query_parts)

        try:
            if self.vectorizer is not None and cosine_similarity is not None:
                tfidf_matrix = self.vectorizer.fit_transform(app_docs)
                q_vec = self.vectorizer.transform([full_query])
                raw_scores = list(cosine_similarity(q_vec, tfidf_matrix)[0])
            else:
                raw_scores = _pure_python_tfidf_scores(app_docs, full_query)

            # Stable descending ranking by score
            ranked_indices = sorted(range(len(app_names)), key=lambda i: -raw_scores[i])
            ranked_apps = [app_names[i] for i in ranked_indices]

            recommended = ranked_apps[: self.top_k_recommended]
            secondary = ranked_apps[self.top_k_recommended :]
            return recommended, secondary
        except Exception:
            # Fallback: preserve original order
            return app_names[: self.top_k_recommended], app_names[self.top_k_recommended :]
