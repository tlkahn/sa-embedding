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

Optional preprocessing with ByT5-Sanskrit:
- Word segmentation (sandhi splitting)
- Lemmatization
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from aksharamukha import transliterate
import torch

# ByT5-Sanskrit imports (optional - for segmentation/lemmatization)
try:
    from transformers import T5ForConditionalGeneration, AutoTokenizer
    BYT5_AVAILABLE = True
except ImportError:
    BYT5_AVAILABLE = False

# GPU support
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


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

# Devanagari similarity pairs (same concepts as IAST, transliterated)
SIMILARITY_PAIRS_DEVANAGARI = [
    # Same concept, different phrasing (English + Devanagari)
    ("prāṇāyāma breath control", "ऊर्ध्वे प्राणो ह्यधो जीवो"),
    ("meditation on emptiness", "निराधारं मनः कृत्वा विकल्पान् न विकल्पयेत्"),
    ("pure consciousness bliss", "चिदानन्दघनात्मानं सर्वस्थानेषु भावयेत्"),
]

# Devanagari dissimilar pairs
DISSIMILARITY_PAIRS_DEVANAGARI = [
    ("breath practice prāṇa", "आकाशं विमलं पश्येत्"),
    ("cooking recipe", "चिदानन्दघनात्मानं सर्वस्थानेषु भावयेत्"),
    ("machine learning algorithm", "प्राणापानौ समौ कृत्वा"),
]


# =============================================================================
# Benchmark Classes
# =============================================================================

@dataclass
class ModelConfig:
    name: str
    model_id: str
    prefix: str = ""  # Some models need query prefix (e.g., E5)
    needs_pooling: bool = False  # True for models like MuRIL that need manual pooling


MODELS = [
    ModelConfig("Vyakyarth", "krutrim-ai-labs/Vyakyarth"),
    ModelConfig("LaBSE", "sentence-transformers/LaBSE"),
    ModelConfig("E5-multilingual", "intfloat/multilingual-e5-large", prefix="query: "),
    ModelConfig("MuRIL", "google/muril-base-cased", needs_pooling=True),
    ModelConfig("BGE-M3", "BAAI/bge-m3"),
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
    # Devanagari similarity discrimination
    similarity_scores_devanagari: dict = field(default_factory=dict)
    dissimilarity_scores_devanagari: dict = field(default_factory=dict)
    # ByT5-Sanskrit preprocessing results
    retrieval_mrr_segmented: float = 0.0
    retrieval_mrr_lemmatized: float = 0.0
    retrieval_mrr_seg_lemma: float = 0.0  # segmented + lemmatized
    similarity_scores_segmented: dict = field(default_factory=dict)
    dissimilarity_scores_segmented: dict = field(default_factory=dict)
    similarity_scores_lemmatized: dict = field(default_factory=dict)
    dissimilarity_scores_lemmatized: dict = field(default_factory=dict)


# =============================================================================
# ByT5-Sanskrit Preprocessing (Segmentation & Lemmatization)
# =============================================================================

class ByT5SanskritPreprocessor:
    """
    ByT5-Sanskrit for Sanskrit text preprocessing.

    Uses task prefixes:
    - "S" = Word Segmentation (sandhi splitting)
    - "L" = Lemmatization
    - "M" = Morphosyntactic tagging

    Reference: Nehrdich et al. 2024, "One Model is All You Need: ByT5-Sanskrit"
    """

    def __init__(self, model_id: str = "chronbmm/sanskrit5-multitask", device: str = None):
        if not BYT5_AVAILABLE:
            raise ImportError("transformers library required for ByT5-Sanskrit")

        self.device = device or DEVICE
        print(f"Loading ByT5-Sanskrit: {model_id} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.model_id = model_id

    def _process(self, text: str, task_prefix: str, max_length: int = 512) -> str:
        """Process text with specified task prefix."""
        input_text = f"{task_prefix}{text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def segment(self, text: str) -> str:
        """Segment Sanskrit text (sandhi splitting)."""
        return self._process(text, "S")

    def lemmatize(self, text: str) -> str:
        """Lemmatize Sanskrit text."""
        return self._process(text, "L")

    def segment_and_lemmatize(self, text: str) -> str:
        """Segment then lemmatize Sanskrit text."""
        segmented = self.segment(text)
        return self.lemmatize(segmented)


def preprocess_corpus_with_byt5(
    corpus: list[str],
    preprocessor: ByT5SanskritPreprocessor,
    mode: str = "segment"  # "segment", "lemmatize", or "both"
) -> list[str]:
    """
    Preprocess corpus using ByT5-Sanskrit.

    Args:
        corpus: List of Sanskrit texts
        preprocessor: ByT5SanskritPreprocessor instance
        mode: "segment" for sandhi splitting, "lemmatize" for lemmatization,
              "both" for segmentation followed by lemmatization
    """
    result = []
    for text in corpus:
        # Skip non-Sanskrit text (e.g., English queries)
        if not any(c in text for c in "āīūṛṝḷḹēōṃḥṅñṭḍṇśṣऀ-ॿ"):
            result.append(text)
            continue

        if mode == "segment":
            processed = preprocessor.segment(text)
        elif mode == "lemmatize":
            processed = preprocessor.lemmatize(text)
        elif mode == "both":
            processed = preprocessor.segment_and_lemmatize(text)
        else:
            processed = text
        result.append(processed)
    return result


# =============================================================================
# MuRIL Wrapper (Manual Pooling)
# =============================================================================

class MuRILWrapper:
    """
    Wrapper for MuRIL that provides SentenceTransformer-compatible interface.

    MuRIL (google/muril-base-cased) is not a sentence transformer, so we need
    to manually apply mean pooling to get sentence embeddings.
    """

    def __init__(self, model_id: str = "google/muril-base-cased", device: str = None):
        from transformers import AutoTokenizer, AutoModel

        self.device = device or DEVICE
        print(f"Loading MuRIL with mean pooling: {model_id} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.model_id = model_id
        self._embedding_dim = self.model.config.hidden_size

    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to token embeddings."""
        token_embeddings = model_output[0]  # First element is last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        """Encode sentences to embeddings using mean pooling."""
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)

        embeddings = self._mean_pooling(output, encoded["attention_mask"])
        return embeddings.cpu().numpy()

    def get_sentence_embedding_dimension(self) -> int:
        return self._embedding_dim


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


def load_model(config: ModelConfig):
    """
    Load model and return (model, load_time_seconds). Uses GPU if available.

    Returns a model with .encode() and .get_sentence_embedding_dimension() methods.
    For models that need pooling (e.g., MuRIL), uses a wrapper class.
    """
    start = time.time()
    if config.needs_pooling:
        # Use wrapper with manual mean pooling
        model = MuRILWrapper(config.model_id, device=DEVICE)
    else:
        # Use SentenceTransformer directly
        model = SentenceTransformer(config.model_id, device=DEVICE)
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


def run_benchmark(
    config: ModelConfig,
    byt5_preprocessor: Optional[ByT5SanskritPreprocessor] = None
) -> BenchmarkResult:
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

    # Devanagari similarity discrimination
    print("Computing Devanagari similarity scores...")
    similarity_scores_deva = {}
    for text1, text2 in SIMILARITY_PAIRS_DEVANAGARI:
        score = compute_similarity(model, text1, text2, config.prefix)
        similarity_scores_deva[(text1[:30], text2[:30])] = score
        print(f"  {text1[:25]}... <-> {text2[:15]}...: {score:.3f}")

    print("Computing Devanagari dissimilarity scores...")
    dissimilarity_scores_deva = {}
    for text1, text2 in DISSIMILARITY_PAIRS_DEVANAGARI:
        score = compute_similarity(model, text1, text2, config.prefix)
        dissimilarity_scores_deva[(text1[:30], text2[:30])] = score
        print(f"  {text1[:25]}... <-> {text2[:15]}...: {score:.3f}")

    # ByT5-Sanskrit preprocessing benchmarks
    mrr_segmented = 0.0
    mrr_lemmatized = 0.0
    mrr_seg_lemma = 0.0
    sim_scores_seg = {}
    dissim_scores_seg = {}
    sim_scores_lemma = {}
    dissim_scores_lemma = {}

    if byt5_preprocessor is not None:
        print("\n" + "-" * 60)
        print("ByT5-Sanskrit Preprocessing Benchmarks")
        print("-" * 60)

        # Preprocess corpus with segmentation
        print("Preprocessing corpus with ByT5 segmentation...")
        corpus_segmented = preprocess_corpus_with_byt5(
            SANSKRIT_CORPUS, byt5_preprocessor, mode="segment"
        )
        print("  Sample segmented:")
        for orig, seg in zip(SANSKRIT_CORPUS[:2], corpus_segmented[:2]):
            print(f"    {orig[:40]}...")
            print(f"    → {seg[:40]}...")

        # Preprocess queries with segmentation
        queries_segmented = [
            (preprocess_corpus_with_byt5([q], byt5_preprocessor, mode="segment")[0], indices)
            for q, indices in RETRIEVAL_TEST_CASES
        ]

        # Retrieval with segmented corpus
        print("Evaluating retrieval (segmented corpus)...")
        mrr_segmented, _, _ = evaluate_retrieval(
            model, corpus_segmented, queries_segmented, config.prefix
        )
        print(f"  MRR (segmented): {mrr_segmented:.3f}")

        # Preprocess corpus with lemmatization
        print("Preprocessing corpus with ByT5 lemmatization...")
        corpus_lemmatized = preprocess_corpus_with_byt5(
            SANSKRIT_CORPUS, byt5_preprocessor, mode="lemmatize"
        )
        print("  Sample lemmatized:")
        for orig, lemma in zip(SANSKRIT_CORPUS[:2], corpus_lemmatized[:2]):
            print(f"    {orig[:40]}...")
            print(f"    → {lemma[:40]}...")

        # Preprocess queries with lemmatization
        queries_lemmatized = [
            (preprocess_corpus_with_byt5([q], byt5_preprocessor, mode="lemmatize")[0], indices)
            for q, indices in RETRIEVAL_TEST_CASES
        ]

        # Retrieval with lemmatized corpus
        print("Evaluating retrieval (lemmatized corpus)...")
        mrr_lemmatized, _, _ = evaluate_retrieval(
            model, corpus_lemmatized, queries_lemmatized, config.prefix
        )
        print(f"  MRR (lemmatized): {mrr_lemmatized:.3f}")

        # Preprocess corpus with both segmentation and lemmatization
        print("Preprocessing corpus with ByT5 segmentation + lemmatization...")
        corpus_seg_lemma = preprocess_corpus_with_byt5(
            SANSKRIT_CORPUS, byt5_preprocessor, mode="both"
        )

        queries_seg_lemma = [
            (preprocess_corpus_with_byt5([q], byt5_preprocessor, mode="both")[0], indices)
            for q, indices in RETRIEVAL_TEST_CASES
        ]

        print("Evaluating retrieval (segmented + lemmatized corpus)...")
        mrr_seg_lemma, _, _ = evaluate_retrieval(
            model, corpus_seg_lemma, queries_seg_lemma, config.prefix
        )
        print(f"  MRR (seg+lemma): {mrr_seg_lemma:.3f}")

        # Similarity discrimination with segmented text
        print("Computing similarity scores (segmented)...")
        for text1, text2 in SIMILARITY_PAIRS[:3]:  # Use first 3 pairs
            t1_seg = preprocess_corpus_with_byt5([text1], byt5_preprocessor, mode="segment")[0]
            t2_seg = preprocess_corpus_with_byt5([text2], byt5_preprocessor, mode="segment")[0]
            score = compute_similarity(model, t1_seg, t2_seg, config.prefix)
            sim_scores_seg[(text1[:30], text2[:30])] = score
            print(f"  {text1[:20]}... <-> {text2[:20]}...: {score:.3f}")

        print("Computing dissimilarity scores (segmented)...")
        for text1, text2 in DISSIMILARITY_PAIRS:
            t1_seg = preprocess_corpus_with_byt5([text1], byt5_preprocessor, mode="segment")[0]
            t2_seg = preprocess_corpus_with_byt5([text2], byt5_preprocessor, mode="segment")[0]
            score = compute_similarity(model, t1_seg, t2_seg, config.prefix)
            dissim_scores_seg[(text1[:30], text2[:30])] = score
            print(f"  {text1[:20]}... <-> {text2[:20]}...: {score:.3f}")

        # Similarity discrimination with lemmatized text
        print("Computing similarity scores (lemmatized)...")
        for text1, text2 in SIMILARITY_PAIRS[:3]:
            t1_lemma = preprocess_corpus_with_byt5([text1], byt5_preprocessor, mode="lemmatize")[0]
            t2_lemma = preprocess_corpus_with_byt5([text2], byt5_preprocessor, mode="lemmatize")[0]
            score = compute_similarity(model, t1_lemma, t2_lemma, config.prefix)
            sim_scores_lemma[(text1[:30], text2[:30])] = score
            print(f"  {text1[:20]}... <-> {text2[:20]}...: {score:.3f}")

        print("Computing dissimilarity scores (lemmatized)...")
        for text1, text2 in DISSIMILARITY_PAIRS:
            t1_lemma = preprocess_corpus_with_byt5([text1], byt5_preprocessor, mode="lemmatize")[0]
            t2_lemma = preprocess_corpus_with_byt5([text2], byt5_preprocessor, mode="lemmatize")[0]
            score = compute_similarity(model, t1_lemma, t2_lemma, config.prefix)
            dissim_scores_lemma[(text1[:30], text2[:30])] = score
            print(f"  {text1[:20]}... <-> {text2[:20]}...: {score:.3f}")

    return BenchmarkResult(
        model_name=config.name,
        embedding_dim=model.get_sentence_embedding_dimension(),
        load_time_sec=load_time,
        avg_encode_time_ms=avg_encode_time,
        similarity_scores=similarity_scores,
        dissimilarity_scores=dissimilarity_scores,
        retrieval_mrr=float(mrr),
        retrieval_recall_at_1=float(r1),
        retrieval_recall_at_3=float(r3),
        cross_script_similarity=float(cross_script_avg),
        retrieval_mrr_devanagari=float(mrr_deva),
        retrieval_recall_at_1_devanagari=float(r1_deva),
        retrieval_recall_at_3_devanagari=float(r3_deva),
        transliteration_consistency=transliteration_consistency,
        similarity_scores_devanagari=similarity_scores_deva,
        dissimilarity_scores_devanagari=dissimilarity_scores_deva,
        retrieval_mrr_segmented=float(mrr_segmented),
        retrieval_mrr_lemmatized=float(mrr_lemmatized),
        retrieval_mrr_seg_lemma=float(mrr_seg_lemma),
        similarity_scores_segmented=sim_scores_seg,
        dissimilarity_scores_segmented=dissim_scores_seg,
        similarity_scores_lemmatized=sim_scores_lemma,
        dissimilarity_scores_lemmatized=dissim_scores_lemma,
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

    # Similarity discrimination (IAST)
    print("\n" + "-" * 80)
    print("Similarity Discrimination - IAST (higher = better separation)")

    for r in results:
        sim_avg = np.mean(list(r.similarity_scores.values()))
        dissim_avg = np.mean(list(r.dissimilarity_scores.values()))
        discrimination = sim_avg - dissim_avg
        print(f"  {r.model_name}: sim_avg={sim_avg:.3f}, dissim_avg={dissim_avg:.3f}, "
              f"discrimination={discrimination:.3f}")

    # Similarity discrimination (Devanagari)
    print("\n" + "-" * 80)
    print("Similarity Discrimination - DEVANAGARI (higher = better separation)")

    for r in results:
        if r.similarity_scores_devanagari and r.dissimilarity_scores_devanagari:
            sim_avg = np.mean(list(r.similarity_scores_devanagari.values()))
            dissim_avg = np.mean(list(r.dissimilarity_scores_devanagari.values()))
            discrimination = sim_avg - dissim_avg
            print(f"  {r.model_name}: sim_avg={sim_avg:.3f}, dissim_avg={dissim_avg:.3f}, "
                  f"discrimination={discrimination:.3f}")

    # ByT5-Sanskrit preprocessing results
    has_byt5_results = any(r.retrieval_mrr_segmented > 0 for r in results)
    if has_byt5_results:
        print("\n" + "-" * 80)
        print("ByT5-SANSKRIT PREPROCESSING (segmentation & lemmatization)")
        print("-" * 80)

        print(f"{'MRR (Raw IAST)':<30} ", end="")
        for r in results:
            print(f"{r.retrieval_mrr:<18.3f} ", end="")
        print()

        print(f"{'MRR (Segmented)':<30} ", end="")
        for r in results:
            print(f"{r.retrieval_mrr_segmented:<18.3f} ", end="")
        print()

        print(f"{'MRR (Lemmatized)':<30} ", end="")
        for r in results:
            print(f"{r.retrieval_mrr_lemmatized:<18.3f} ", end="")
        print()

        print(f"{'MRR (Seg+Lemma)':<30} ", end="")
        for r in results:
            print(f"{r.retrieval_mrr_seg_lemma:<18.3f} ", end="")
        print()

        # MRR Delta from preprocessing
        print("\n  MRR Delta from preprocessing (positive = improvement):")
        for r in results:
            seg_delta = r.retrieval_mrr_segmented - r.retrieval_mrr
            lemma_delta = r.retrieval_mrr_lemmatized - r.retrieval_mrr
            both_delta = r.retrieval_mrr_seg_lemma - r.retrieval_mrr
            print(f"  {r.model_name}: seg={seg_delta:+.3f}, lemma={lemma_delta:+.3f}, both={both_delta:+.3f}")

        # Similarity discrimination with preprocessing
        print("\n  Similarity Discrimination - SEGMENTED (higher = better):")
        for r in results:
            if r.similarity_scores_segmented and r.dissimilarity_scores_segmented:
                sim_avg = np.mean(list(r.similarity_scores_segmented.values()))
                dissim_avg = np.mean(list(r.dissimilarity_scores_segmented.values()))
                discrimination = sim_avg - dissim_avg
                print(f"    {r.model_name}: sim={sim_avg:.3f}, dissim={dissim_avg:.3f}, disc={discrimination:.3f}")

        print("\n  Similarity Discrimination - LEMMATIZED (higher = better):")
        for r in results:
            if r.similarity_scores_lemmatized and r.dissimilarity_scores_lemmatized:
                sim_avg = np.mean(list(r.similarity_scores_lemmatized.values()))
                dissim_avg = np.mean(list(r.dissimilarity_scores_lemmatized.values()))
                discrimination = sim_avg - dissim_avg
                print(f"    {r.model_name}: sim={sim_avg:.3f}, dissim={dissim_avg:.3f}, disc={discrimination:.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sanskrit Embedding Models Benchmark")
    parser.add_argument("--byt5", action="store_true",
                        help="Enable ByT5-Sanskrit preprocessing (segmentation/lemmatization)")
    parser.add_argument("--byt5-model", type=str, default="chronbmm/sanskrit5-multitask",
                        help="ByT5-Sanskrit model ID (default: chronbmm/sanskrit5-multitask)")
    args = parser.parse_args()

    print("Sanskrit Embedding Models Benchmark")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Corpus size: {len(SANSKRIT_CORPUS)} sentences")
    print(f"Retrieval test cases: {len(RETRIEVAL_TEST_CASES)}")
    print(f"Similarity pairs: {len(SIMILARITY_PAIRS)}")
    print(f"Dissimilarity pairs: {len(DISSIMILARITY_PAIRS)}")

    # Load ByT5-Sanskrit preprocessor if requested
    byt5_preprocessor = None
    if args.byt5:
        if BYT5_AVAILABLE:
            print(f"\nLoading ByT5-Sanskrit preprocessor: {args.byt5_model}")
            byt5_preprocessor = ByT5SanskritPreprocessor(model_id=args.byt5_model)
            print("ByT5-Sanskrit loaded successfully.")
        else:
            print("\nWarning: ByT5-Sanskrit requested but transformers not available.")
            print("Install with: pip install transformers")

    results = []
    for config in MODELS:
        try:
            result = run_benchmark(config, byt5_preprocessor=byt5_preprocessor)
            results.append(result)
        except Exception as e:
            print(f"\nError benchmarking {config.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if results:
        print_comparison_table(results)

    print("\n" + "=" * 80)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
