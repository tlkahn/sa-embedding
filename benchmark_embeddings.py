"""
Sanskrit Embedding Models Benchmark

Compares semantic search performance across:
- Vyakyarth (krutrim-ai-labs/Vyakyarth) - Indic-optimized
- LaBSE (sentence-transformers/LaBSE) - Cross-lingual
- multilingual-e5-large (intfloat/multilingual-e5-large) - Strong multilingual baseline

Metrics:
- Embedding latency
- Semantic similarity consistency
- Retrieval accuracy (MRR, Recall@k)
- Cross-lingual retrieval (English → Sanskrit)
- Script comparison (IAST vs Devanagari via transliteration)
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from aksharamukha import transliterate


# =============================================================================
# Test Data: Sanskrit Corpus and Queries
# =============================================================================

# Sanskrit sentences (IAST) - from Vijñānabhairava and related texts
SANSKRIT_CORPUS = [
    # Breath/prāṇa related
    "ūrdhve prāṇo hy adho jīvo visargātmā paroccaret",
    "prāṇāpānau samau kṛtvā nāsābhyantaracāriṇau",
    "madhye vihāya śūnyaṃ tu prāṇo jīvaḥ prakīrtitaḥ",
    "prāṇasya śūnyapadavī tadā bhairavaṃ vapuḥ",
    # Meditation/dhyāna related
    "dhyāyec chaktisvarūpaṃ yaḥ śaktyāveśaḥ sa ucyate",
    "vyāpti sarvātmanā jñeyā sāvayavā niravayavā",
    "bhāvanāṃ bhāvayed yastu śaktiṃ tāṃ pratipadyate",
    # Consciousness/cit related
    "cidānandaghanātmānaṃ sarvasthāneṣu bhāvayet",
    "sarvataḥ svātmanaḥ pūrṇaṃ bhāvayed bhāvanātmanaḥ",
    "ananyacetāḥ satataṃ nāmaskāram ekam āśrayet",
    # Space/ākāśa related
    "nirādhāraṃ manaḥ kṛtvā vikalpān na vikalpayet",
    "ākāśaṃ vimalaṃ paśyen nirmalātmā prajāyate",
    "tanmayatvena manaḥ kṛtvā jagat paśyati kevalīm",
]

# Devanagari versions of some sentences
SANSKRIT_CORPUS_DEVANAGARI = [
    "ऊर्ध्वे प्राणो ह्यधो जीवो विसर्गात्मा परोच्चरेत्",
    "प्राणापानौ समौ कृत्वा नासाभ्यन्तरचारिणौ",
    "ध्यायेच्छक्तिस्वरूपं यः शक्त्यावेशः स उच्यते",
    "चिदानन्दघनात्मानं सर्वस्थानेषु भावयेत्",
    "आकाशं विमलं पश्येन्निर्मलात्मा प्रजायते",
]

# Ground truth query-document pairs for retrieval evaluation
# Format: (query, relevant_doc_indices)
RETRIEVAL_TEST_CASES = [
    # English queries → Sanskrit documents
    ("breath practice ascending and descending", [0, 1, 2, 3]),
    ("meditation on consciousness", [4, 5, 6, 7, 8]),
    ("contemplation of space and emptiness", [10, 11, 12]),
    ("energy and power visualization", [4, 6]),
    # Sanskrit queries (IAST)
    ("prāṇa apāna dhāraṇā", [0, 1, 2, 3]),
    ("śakti bhāvanā dhyāna", [4, 5, 6]),
    ("cit ānanda svarūpa", [7, 8, 9]),
]

# Semantic similarity pairs (should have high similarity)
SIMILARITY_PAIRS = [
    # Same concept, different phrasing
    ("prāṇāyāma breath control", "ūrdhve prāṇo hy adho jīvo"),
    ("meditation on emptiness", "nirādhāraṃ manaḥ kṛtvā vikalpān na vikalpayet"),
    ("pure consciousness bliss", "cidānandaghanātmānaṃ sarvasthāneṣu bhāvayet"),
    # Cross-script (IAST vs Devanagari)
    ("ūrdhve prāṇo hy adho jīvo", "ऊर्ध्वे प्राणो ह्यधो जीवो"),
    ("dhyāyec chaktisvarūpaṃ", "ध्यायेच्छक्तिस्वरूपं यः"),
]

# Dissimilar pairs (should have low similarity)
DISSIMILARITY_PAIRS = [
    ("breath practice prāṇa", "ākāśaṃ vimalaṃ paśyen"),
    ("cooking recipe", "cidānandaghanātmānaṃ sarvasthāneṣu bhāvayet"),
    ("machine learning algorithm", "prāṇāpānau samau kṛtvā"),
]


# =============================================================================
# Benchmark Classes
# =============================================================================

@dataclass
class ModelConfig:
    name: str
    model_id: str
    prefix: str = ""  # Some models need query prefix (e.g., E5)


MODELS = [
    ModelConfig("Vyakyarth", "krutrim-ai-labs/Vyakyarth"),
    ModelConfig("LaBSE", "sentence-transformers/LaBSE"),
    ModelConfig("E5-multilingual", "intfloat/multilingual-e5-large", prefix="query: "),
]


@dataclass
class BenchmarkResult:
    model_name: str
    embedding_dim: int
    load_time_sec: float
    avg_encode_time_ms: float
    similarity_scores: dict  # pair -> score
    dissimilarity_scores: dict
    retrieval_mrr: float
    retrieval_recall_at_1: float
    retrieval_recall_at_3: float
    cross_script_similarity: float
    # Transliteration benchmark results (Devanagari)
    retrieval_mrr_devanagari: float = 0.0
    retrieval_recall_at_1_devanagari: float = 0.0
    retrieval_recall_at_3_devanagari: float = 0.0
    transliteration_consistency: float = 0.0  # IAST embedding vs transliterated Devanagari embedding


# =============================================================================
# Transliteration Functions
# =============================================================================

def transliterate_iast_to_devanagari(text: str) -> str:
    """Transliterate IAST text to Devanagari using aksharamukha."""
    return transliterate.process("IAST", "Devanagari", text)


def transliterate_corpus(corpus: list[str]) -> list[str]:
    """Transliterate entire corpus from IAST to Devanagari."""
    return [transliterate_iast_to_devanagari(text) for text in corpus]


def transliterate_queries(test_cases: list[tuple[str, list[int]]]) -> list[tuple[str, list[int]]]:
    """
    Transliterate Sanskrit queries to Devanagari.
    English queries are kept as-is.
    """
    result = []
    for query, indices in test_cases:
        # Check if query contains Sanskrit (IAST diacritics)
        if any(c in query for c in "āīūṛṝḷḹēōṃḥṅñṭḍṇśṣ"):
            result.append((transliterate_iast_to_devanagari(query), indices))
        else:
            # English query - keep as-is
            result.append((query, indices))
    return result


def load_model(config: ModelConfig) -> tuple[SentenceTransformer, float]:
    """Load model and return (model, load_time_seconds)."""
    start = time.time()
    model = SentenceTransformer(config.model_id)
    load_time = time.time() - start
    return model, load_time


def benchmark_encoding_speed(
    model: SentenceTransformer,
    texts: list[str],
    n_runs: int = 3
) -> float:
    """Return average encoding time in milliseconds per text."""
    times = []
    for _ in range(n_runs):
        start = time.time()
        model.encode(texts)
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed / len(texts))
    return np.mean(times)


def compute_similarity(
    model: SentenceTransformer,
    text1: str,
    text2: str,
    prefix: str = ""
) -> float:
    """Compute cosine similarity between two texts."""
    emb1 = model.encode([prefix + text1])
    emb2 = model.encode([prefix + text2])
    return float(cosine_similarity(emb1, emb2)[0, 0])


def evaluate_retrieval(
    model: SentenceTransformer,
    corpus: list[str],
    test_cases: list[tuple[str, list[int]]],
    prefix: str = ""
) -> tuple[float, float, float]:
    """
    Evaluate retrieval performance.
    Returns: (MRR, Recall@1, Recall@3)
    """
    corpus_embeddings = model.encode(corpus)

    mrr_scores = []
    recall_at_1 = []
    recall_at_3 = []

    for query, relevant_indices in test_cases:
        query_emb = model.encode([prefix + query])
        similarities = cosine_similarity(query_emb, corpus_embeddings)[0]
        ranked_indices = np.argsort(similarities)[::-1]

        # MRR: reciprocal rank of first relevant document
        for rank, idx in enumerate(ranked_indices, 1):
            if idx in relevant_indices:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

        # Recall@1: is top result relevant?
        recall_at_1.append(1.0 if ranked_indices[0] in relevant_indices else 0.0)

        # Recall@3: any of top 3 relevant?
        top_3 = set(ranked_indices[:3])
        recall_at_3.append(1.0 if top_3 & set(relevant_indices) else 0.0)

    return np.mean(mrr_scores), np.mean(recall_at_1), np.mean(recall_at_3)


def run_benchmark(config: ModelConfig) -> BenchmarkResult:
    """Run full benchmark for a single model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config.name}")
    print(f"Model ID: {config.model_id}")
    print(f"{'='*60}")

    # Load model
    print("Loading model...")
    model, load_time = load_model(config)
    print(f"  Load time: {load_time:.2f}s")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Encoding speed
    print("Measuring encoding speed...")
    avg_encode_time = benchmark_encoding_speed(model, SANSKRIT_CORPUS)
    print(f"  Avg encode time: {avg_encode_time:.2f}ms per text")

    # Similarity pairs
    print("Computing similarity scores...")
    similarity_scores = {}
    for text1, text2 in SIMILARITY_PAIRS:
        score = compute_similarity(model, text1, text2, config.prefix)
        similarity_scores[(text1[:30], text2[:30])] = score
        print(f"  {text1[:25]}... <-> {text2[:25]}...: {score:.3f}")

    # Dissimilarity pairs
    print("Computing dissimilarity scores...")
    dissimilarity_scores = {}
    for text1, text2 in DISSIMILARITY_PAIRS:
        score = compute_similarity(model, text1, text2, config.prefix)
        dissimilarity_scores[(text1[:30], text2[:30])] = score
        print(f"  {text1[:25]}... <-> {text2[:25]}...: {score:.3f}")

    # Cross-script similarity (IAST vs Devanagari)
    print("Evaluating cross-script consistency...")
    cross_script_scores = []
    for iast, deva in zip(SANSKRIT_CORPUS[:5], SANSKRIT_CORPUS_DEVANAGARI):
        score = compute_similarity(model, iast, deva, config.prefix)
        cross_script_scores.append(score)
        print(f"  IAST<->Devanagari: {score:.3f}")
    cross_script_avg = np.mean(cross_script_scores)
    print(f"  Average cross-script similarity: {cross_script_avg:.3f}")

    # Retrieval evaluation (IAST)
    print("Evaluating retrieval performance (IAST corpus)...")
    mrr, r1, r3 = evaluate_retrieval(model, SANSKRIT_CORPUS, RETRIEVAL_TEST_CASES, config.prefix)
    print(f"  MRR: {mrr:.3f}")
    print(f"  Recall@1: {r1:.3f}")
    print(f"  Recall@3: {r3:.3f}")

    # Transliteration benchmark (Devanagari)
    print("Transliterating corpus to Devanagari...")
    corpus_devanagari = transliterate_corpus(SANSKRIT_CORPUS)
    queries_devanagari = transliterate_queries(RETRIEVAL_TEST_CASES)

    print("Evaluating retrieval performance (Devanagari corpus)...")
    mrr_deva, r1_deva, r3_deva = evaluate_retrieval(
        model, corpus_devanagari, queries_devanagari, config.prefix
    )
    print(f"  MRR: {mrr_deva:.3f}")
    print(f"  Recall@1: {r1_deva:.3f}")
    print(f"  Recall@3: {r3_deva:.3f}")

    # Transliteration consistency: compare IAST vs transliterated Devanagari embeddings
    print("Evaluating transliteration consistency...")
    consistency_scores = []
    for iast_text, deva_text in zip(SANSKRIT_CORPUS, corpus_devanagari):
        score = compute_similarity(model, iast_text, deva_text, config.prefix)
        consistency_scores.append(score)
    transliteration_consistency = float(np.mean(consistency_scores))
    print(f"  Avg IAST<->Transliterated Devanagari similarity: {transliteration_consistency:.3f}")

    return BenchmarkResult(
        model_name=config.name,
        embedding_dim=model.get_sentence_embedding_dimension(),
        load_time_sec=load_time,
        avg_encode_time_ms=avg_encode_time,
        similarity_scores=similarity_scores,
        dissimilarity_scores=dissimilarity_scores,
        retrieval_mrr=mrr,
        retrieval_recall_at_1=r1,
        retrieval_recall_at_3=r3,
        cross_script_similarity=cross_script_avg,
        retrieval_mrr_devanagari=mrr_deva,
        retrieval_recall_at_1_devanagari=r1_deva,
        retrieval_recall_at_3_devanagari=r3_deva,
        transliteration_consistency=transliteration_consistency,
    )


def print_comparison_table(results: list[BenchmarkResult]):
    """Print side-by-side comparison of all models."""
    print("\n")
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Header
    print(f"\n{'Metric':<30} ", end="")
    for r in results:
        print(f"{r.model_name:<18} ", end="")
    print()
    print("-" * 80)

    # Metrics
    print(f"{'Embedding Dimension':<30} ", end="")
    for r in results:
        print(f"{r.embedding_dim:<18} ", end="")
    print()

    print(f"{'Load Time (sec)':<30} ", end="")
    for r in results:
        print(f"{r.load_time_sec:<18.2f} ", end="")
    print()

    print(f"{'Encode Time (ms/text)':<30} ", end="")
    for r in results:
        print(f"{r.avg_encode_time_ms:<18.2f} ", end="")
    print()

    print(f"{'Cross-Script Similarity':<30} ", end="")
    for r in results:
        print(f"{r.cross_script_similarity:<18.3f} ", end="")
    print()

    print(f"{'Retrieval MRR':<30} ", end="")
    for r in results:
        print(f"{r.retrieval_mrr:<18.3f} ", end="")
    print()

    print(f"{'Retrieval Recall@1':<30} ", end="")
    for r in results:
        print(f"{r.retrieval_recall_at_1:<18.3f} ", end="")
    print()

    print(f"{'Retrieval Recall@3':<30} ", end="")
    for r in results:
        print(f"{r.retrieval_recall_at_3:<18.3f} ", end="")
    print()

    # Devanagari retrieval results
    print("\n" + "-" * 80)
    print("DEVANAGARI CORPUS (via aksharamukha transliteration)")
    print("-" * 80)

    print(f"{'Retrieval MRR (Devanagari)':<30} ", end="")
    for r in results:
        print(f"{r.retrieval_mrr_devanagari:<18.3f} ", end="")
    print()

    print(f"{'Retrieval R@1 (Devanagari)':<30} ", end="")
    for r in results:
        print(f"{r.retrieval_recall_at_1_devanagari:<18.3f} ", end="")
    print()

    print(f"{'Retrieval R@3 (Devanagari)':<30} ", end="")
    for r in results:
        print(f"{r.retrieval_recall_at_3_devanagari:<18.3f} ", end="")
    print()

    print(f"{'Transliteration Consistency':<30} ", end="")
    for r in results:
        print(f"{r.transliteration_consistency:<18.3f} ", end="")
    print()

    # Delta between IAST and Devanagari
    print("\n" + "-" * 80)
    print("IAST vs Devanagari Delta (positive = IAST better)")
    print("-" * 80)

    print(f"{'MRR Delta':<30} ", end="")
    for r in results:
        delta = r.retrieval_mrr - r.retrieval_mrr_devanagari
        print(f"{delta:<+18.3f} ", end="")
    print()

    print(f"{'R@1 Delta':<30} ", end="")
    for r in results:
        delta = r.retrieval_recall_at_1 - r.retrieval_recall_at_1_devanagari
        print(f"{delta:<+18.3f} ", end="")
    print()

    # Similarity discrimination
    print("\n" + "-" * 80)
    print("Similarity Discrimination (higher = better separation)")

    for r in results:
        sim_avg = np.mean(list(r.similarity_scores.values()))
        dissim_avg = np.mean(list(r.dissimilarity_scores.values()))
        discrimination = sim_avg - dissim_avg
        print(f"  {r.model_name}: sim_avg={sim_avg:.3f}, dissim_avg={dissim_avg:.3f}, "
              f"discrimination={discrimination:.3f}")


def main():
    print("Sanskrit Embedding Models Benchmark")
    print("=" * 60)
    print(f"Corpus size: {len(SANSKRIT_CORPUS)} sentences")
    print(f"Retrieval test cases: {len(RETRIEVAL_TEST_CASES)}")
    print(f"Similarity pairs: {len(SIMILARITY_PAIRS)}")
    print(f"Dissimilarity pairs: {len(DISSIMILARITY_PAIRS)}")

    results = []
    for config in MODELS:
        try:
            result = run_benchmark(config)
            results.append(result)
        except Exception as e:
            print(f"\nError benchmarking {config.name}: {e}")
            continue

    if results:
        print_comparison_table(results)

    print("\n" + "=" * 80)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
