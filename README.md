# Sanskrit Text Embeddings for Semantic Search: A Practical Survey

**Author's Note**: This survey synthesizes available information as of early 2025. The field of Indic NLP is evolving rapidly, and some claims—particularly from industry releases without peer-reviewed papers—should be treated with appropriate caution.

---

## 1. Introduction

Building semantic search over Sanskrit texts presents unique challenges absent in modern language NLP:

- **Sandhi**: Phonological fusion at word boundaries obscures token boundaries
- **Compounding**: Extensive nominal composition (समास) creates novel lexical units
- **Morphological richness**: Eight cases, three numbers, complex verbal system
- **Script variation**: Texts exist in Devanagari, IAST romanization, and regional scripts
- **Limited training data**: Sanskrit web presence is minuscule relative to corpus requirements for modern LLMs

This survey examines available embedding approaches for Sanskrit semantic search, with particular attention to sentence-level retrieval suitable for systems like pgvector.

---

## 2. Tokenization: The Upstream Problem

Before embeddings, tokenization determines how text is segmented. Standard BPE tokenizers (tiktoken, SentencePiece trained on English) perform poorly on Sanskrit:

| Input | tiktoken (cl100k) | Tokens | Chars/Token |
|-------|-------------------|--------|-------------|
| "prāṇa" (IAST) | ["pr", "ā", "ṇ", "a"] | ~4 | ~1.5 |
| "प्राण" (Devanagari) | ["प्रा", "ण"] | ~2-3 | ~2 |
| "breath" (English) | ["breath"] | 1 | 6 |

**Implications**:
- API costs scale with token count
- Semantic coherence degrades when morphemes fragment
- Cross-lingual alignment suffers

**Recommendations**:
- For OpenAI/Anthropic APIs: Prefer Devanagari over IAST (marginally better tokenization)
- For local models: Use Indic-specific tokenizers (Krutrim, IndicBERT)
- Byte-level models (ByT5-Sanskrit) excel at linguistic analysis but do not improve retrieval quality when used for preprocessing (see Section 5.10)

---

## 3. Embedding Models: A Taxonomy

### 3.1 Sanskrit-Specific Models

| Model | Architecture | Training Data | Sentence-Native | Paper | Notes |
|-------|--------------|---------------|-----------------|-------|-------|
| **ByT5-Sanskrit** | Byte-level T5 | Sanskrit corpora | ❌ (task-specific) | EMNLP 2024 | SOTA for segmentation, parsing, OCR correction |
| **albert-base-sanskrit** | ALBERT | Sanskrit Wikipedia | ❌ (needs pooling) | ❌ | Devanagari only; small training corpus |
| **FastText Sanskrit** | Skip-gram + char n-grams | CommonCrawl | ❌ (word-level) | ✅ (Facebook) | Good OOV handling; domain mismatch for classical texts |

**ByT5-Sanskrit** (Nehrdich, Hellwig, Keutzer — EMNLP 2024 Findings) represents current SOTA for Sanskrit NLP tasks. Its byte-level architecture elegantly handles sandhi and script variation. However, it is optimized for morphological analysis rather than sentence similarity.

*Gap*: No dedicated Sanskrit sentence-transformer exists. This is a genuine lacuna in the literature.

### 3.2 Multilingual Models with Sanskrit Support

| Model | Sanskrit Coverage | Sentence-Native | Params | Paper | Provenance |
|-------|-------------------|-----------------|--------|-------|------------|
| **MuRIL** | ✅ Explicit (17 langs) | ❌ (needs pooling) | 236M | arXiv:2103.10730 | Google Research India |
| **Vyakyarth** | ✅ Explicit (10 Indic) | ✅ Native | 278M | ❌ (under review) | Krutrim AI (Ola) |
| **IndicBERT** | ❌ (12 langs, no Sanskrit) | ❌ | 18M | ✅ | AI4Bharat |
| **LaBSE** | ✅ (109 langs incl. Indic) | ✅ Native | 470M | ✅ (Google) | Google |
| **LASER3** | ✅ (200 langs, 27 Indic) | ✅ Native | — | ✅ (Meta) | Meta AI |

**MuRIL** remains the academically defensible choice with peer-reviewed methodology. For sentence embeddings, use `pooled_output` (CLS token) or mean pooling, though neither is optimized for retrieval. Community fine-tunes exist (`sbastola/muril-base-cased-sentence-transformer-snli`, `pushpdeep/sbert-en_hi-muril`) but lack Sanskrit-specific evaluation.

**Vyakyarth** claims superior performance on Indic retrieval benchmarks (97.8 avg on FLORES vs. Jina-v3's 96.0 per BharatBench). Built on XLM-R-Multilingual with contrastive fine-tuning. However:
- No peer-reviewed paper (cited as "under review")
- Benchmarked on modern Indic, not classical Sanskrit
- Krutrim Community License (not MIT/Apache)

*Recommendation*: For academic work requiring citation, use MuRIL + mean pooling. For production systems prioritizing quality, benchmark Vyakyarth against your specific corpus.

### 3.3 General Multilingual Models

| Model | Sanskrit Support | Sentence-Native | Dim | Notes |
|-------|------------------|-----------------|-----|-------|
| **intfloat/multilingual-e5-large** | Implicit (IAST exposure) | ✅ | 1024 | Strong general multilingual; handles IAST reasonably |
| **OpenAI text-embedding-3-large** | Implicit | ✅ | 3072 | 54.9% on MIRACL multilingual; expensive |
| **sentence-transformers/LaBSE** | ✅ Indic coverage | ✅ | 768 | Good cross-lingual retrieval baseline |
| **paraphrase-multilingual-MiniLM-L12-v2** | Limited | ✅ | 384 | Fast but weak on Indic |

For IAST-romanized texts, **multilingual-e5-large** often performs surprisingly well due to exposure to Indological literature in training data. Worth benchmarking before committing to Indic-specific models.

---

## 4. The FastText Alternative

A 2022 LREC paper on Buddhist Sanskrit embeddings (Lugli et al.) found that **fastText outperformed BERT for semantic similarity**, while BERT excelled at word analogy tasks. This counterintuitive result deserves attention.

**Why FastText works for Sanskrit**:
1. Character n-gram subwords capture morphological regularity
2. No tokenization preprocessing required (handles OOV gracefully)
3. Efficient training on domain-specific corpora
4. Particularly effective for rare forms and OCR errors

**Practical setup**:
```python
import fasttext
import numpy as np

# Pre-trained CommonCrawl Sanskrit
ft = fasttext.load_model('cc.sa.300.bin')

def sentence_embedding(text, model):
    words = text.split()  # Requires pre-segmented input
    vectors = [model.get_word_vector(w) for w in words]
    return np.mean(vectors, axis=0)
```

**Caveats**:
- Pre-trained vectors trained on modern web Sanskrit (Wikipedia, religious sites)
- Domain mismatch for classical/tantric texts
- Works best with pre-segmented text, but note that segmentation may not help transformer-based embeddings (see Section 5.10)

*Recommendation for specialized corpora*: Train custom fastText on domain texts (DCS, GRETIL śaiva corpus) rather than using CommonCrawl vectors:

```bash
./fasttext skipgram -input your_sanskrit_corpus.txt -output sanskrit_custom -dim 300 -minCount 2
```

---

## 5. Empirical Benchmark: IAST vs Devanagari Script Performance

To validate the theoretical recommendations above, we conducted systematic benchmarks comparing three sentence embedding models across script variants. The benchmark uses `aksharamukha` for IAST→Devanagari transliteration to ensure consistent test data.

### 5.1 Model Selection Rationale

We selected three models representing distinct approaches to multilingual embeddings, specifically chosen to test hypotheses about Sanskrit encoding:

| Model | Category | Why Selected |
|-------|----------|--------------|
| **Vyakyarth** | Indic-optimized | Tests whether Indic-specific training improves Sanskrit retrieval. Claims 97.8 avg on FLORES (vs. Jina-v3's 96.0). Built on XLM-R with contrastive fine-tuning on 10 Indic languages. |
| **LaBSE** | Cross-lingual | Google's language-agnostic model (109 languages). Peer-reviewed (ACL 2022). Designed specifically for cross-lingual retrieval — ideal for Sanskrit↔English queries. Explicit Indic coverage. |
| **E5-multilingual** | General multilingual | Strong baseline representing "no Indic-specific training" approach. Tests whether large-scale multilingual pretraining on general web data includes sufficient Sanskrit/IAST exposure. |

**Selection criteria**:

1. **Sentence-native embeddings**: All three models produce sentence embeddings directly without requiring manual pooling. This ensures fair comparison and practical deployment.

2. **Open weights**: All models are freely available on HuggingFace, enabling reproducibility and local deployment (important for potentially sensitive manuscript work).

3. **Distinct training paradigms**:
   - Vyakyarth: Indic-focused contrastive learning
   - LaBSE: Dual-encoder trained for cross-lingual retrieval
   - E5: Large-scale weakly-supervised contrastive learning

4. **Practical considerations**: All run efficiently on consumer hardware (768–1024 dim), have active maintenance, and represent production-viable options.

**Models evaluated (added based on recommendations)**:

| Model | Status | Finding |
|-------|--------|---------|
| **MuRIL** | ✅ Added | **Unsuitable for retrieval** — negative discrimination (-0.005), clusters all text at ~0.99 similarity |
| **BGE-M3** | ✅ Added | Excellent recall (1.0 Recall@3), mediocre MRR (0.643). Good for recall-focused applications |

**Models not included**:

| Model | Reason for Exclusion |
|-------|---------------------|
| **OpenAI text-embedding-3** | Commercial API, not open weights. Cost prohibitive for large corpus indexing. |
| **LASER3** | Meta's model; 200 languages but different architecture (encoder-decoder). |
| **Jina v3** | Commercial model with free tier. Limited peer validation for Indic claims. |
| **IndicSBERT** | Community fine-tune; limited documentation and uncertain maintenance. |

**Key finding from expanded benchmark**: MuRIL, despite being the academically-cited standard for Indic NLP, is **unsuitable for Sanskrit semantic search**. Its mean-pooled embeddings cluster all text at ~0.99 similarity regardless of semantic content. LaBSE should be used instead for citable, peer-reviewed results.

### 5.2 Benchmark Methodology

**Models tested**:
- **Vyakyarth** (`krutrim-ai-labs/Vyakyarth`) — Indic-optimized, 768-dim
- **LaBSE** (`sentence-transformers/LaBSE`) — Cross-lingual, 768-dim
- **E5-multilingual** (`intfloat/multilingual-e5-large`) — General multilingual, 1024-dim
- **MuRIL** (`google/muril-base-cased`) — Academic standard with mean pooling, 768-dim
- **BGE-M3** (`BAAI/bge-m3`) — Recent SOTA multilingual, 1024-dim

**Test corpus**: 13 Sanskrit sentences from Vijñānabhairava and related tantric texts, covering four semantic categories:
- Breath/prāṇa practices (4 sentences)
- Meditation/dhyāna techniques (3 sentences)
- Consciousness/cit contemplations (3 sentences)
- Space/ākāśa visualizations (3 sentences)

**Metrics**:
- **Retrieval**: MRR (Mean Reciprocal Rank), Recall@1, Recall@3 on 7 query-document test cases
- **Similarity Discrimination**: Difference between average similarity of semantically related pairs vs. unrelated pairs (higher = better separation)
- **Transliteration Consistency**: Cosine similarity between IAST and aksharamukha-transliterated Devanagari embeddings
- **Encoding Speed**: Milliseconds per sentence

### 5.3 Retrieval Performance Results

#### IAST Corpus

| Metric | Vyakyarth | LaBSE | E5-multilingual | MuRIL | BGE-M3 |
|--------|-----------|-------|-----------------|-------|--------|
| MRR | 0.509 | **0.759** | 0.714 | 0.605 | 0.643 |
| Recall@1 | 0.286 | **0.714** | 0.571 | 0.429 | 0.429 |
| Recall@3 | 0.571 | 0.714 | 0.857 | 0.714 | **1.000** |

#### Devanagari Corpus (via aksharamukha transliteration)

| Metric | Vyakyarth | LaBSE | E5-multilingual | MuRIL | BGE-M3 |
|--------|-----------|-------|-----------------|-------|--------|
| MRR | 0.929 | 0.929 | **1.000** | 0.726 | 0.857 |
| Recall@1 | 0.857 | 0.857 | **1.000** | 0.571 | 0.714 |
| Recall@3 | **1.000** | **1.000** | **1.000** | 0.857 | **1.000** |

#### Script Impact (Devanagari − IAST)

| Metric | Vyakyarth | LaBSE | E5-multilingual | MuRIL | BGE-M3 |
|--------|-----------|-------|-----------------|-------|--------|
| ΔMRR | **+0.420** | +0.170 | +0.286 | +0.121 | +0.214 |
| ΔRecall@1 | **+0.571** | +0.143 | +0.429 | +0.143 | +0.286 |

**Key findings**:
1. **LaBSE achieves best IAST MRR** (0.759), confirming its effectiveness for Sanskrit retrieval
2. **E5-multilingual achieves perfect Devanagari retrieval** (1.0 MRR) despite no explicit Sanskrit training
3. **MuRIL underperforms** despite being the academic standard — its mean-pooled embeddings lack retrieval optimization
4. **BGE-M3 excels at recall** (1.0 Recall@3 on IAST) but has mediocre precision (0.643 MRR)
5. **All models improve with Devanagari**, with Vyakyarth showing the largest gain (+0.420 MRR)

### 5.4 Similarity Discrimination

Discrimination measures a model's ability to assign high similarity to semantically related pairs and low similarity to unrelated pairs. Higher values indicate better semantic separation.

#### IAST Pairs

| Model | Similar Avg | Dissimilar Avg | Discrimination |
|-------|-------------|----------------|----------------|
| LaBSE | 0.378 | 0.258 | **0.120** |
| BGE-M3 | 0.499 | 0.398 | 0.101 |
| Vyakyarth | 0.328 | 0.244 | 0.084 |
| E5-multilingual | 0.830 | 0.784 | 0.046 |
| MuRIL | 0.987 | 0.992 | **-0.005** |

#### Devanagari Pairs

| Model | Similar Avg | Dissimilar Avg | Discrimination |
|-------|-------------|----------------|----------------|
| LaBSE | 0.330 | 0.094 | **0.236** |
| Vyakyarth | 0.441 | 0.245 | 0.196 |
| BGE-M3 | 0.417 | 0.367 | 0.050 |
| E5-multilingual | 0.774 | 0.756 | 0.017 |
| MuRIL | 0.988 | 0.987 | 0.001 |

**Key findings**:

1. **LaBSE has strongest discrimination** in both scripts, with Devanagari nearly doubling IAST performance (0.236 vs 0.120)
2. **MuRIL has catastrophic discrimination failure** — negative discrimination (-0.005) means it assigns *higher* similarity to unrelated pairs than related pairs. With sim_avg=0.987 and dissim_avg=0.992, everything clusters at near-1.0 similarity, making semantic search impossible.
3. **BGE-M3 is second-best on IAST** (0.101) but degrades on Devanagari (0.050)
4. **Vyakyarth improves 2.3× with Devanagari** (0.196 vs 0.084), confirming native script preference
5. **E5-multilingual clusters too tightly** — dissimilar pairs score 0.756–0.784, but still better than MuRIL
6. **LaBSE's dissimilar scores drop to 0.094 in Devanagari** — it correctly identifies unrelated content as genuinely different

**Critical warning about MuRIL**: Despite being the academically-cited standard for Indic NLP, MuRIL with mean pooling is **unsuitable for Sanskrit semantic search**. Its embeddings cluster all text at ~0.99 similarity regardless of semantic content. This is likely because MuRIL was trained for classification tasks (MLM, NSP), not sentence similarity. Use LaBSE instead.

### 5.5 Transliteration Consistency

This measures whether a model produces similar embeddings for the same text in different scripts (IAST vs Devanagari).

| Model | Consistency Score | Interpretation |
|-------|-------------------|----------------|
| MuRIL | **0.987** | Near-identical (but meaningless — see below) |
| E5-multilingual | 0.901 | Near-identical embeddings across scripts |
| BGE-M3 | 0.573 | Moderate script sensitivity |
| LaBSE | 0.457 | Moderate script sensitivity |
| Vyakyarth | 0.343 | High script sensitivity |

**Implications**:

- **MuRIL's high consistency (0.987) is misleading** — it reflects the model's tendency to cluster everything at ~0.99 similarity, not genuine script understanding
- **E5-multilingual** treats IAST and Devanagari as essentially the same text — useful for script-agnostic search
- **BGE-M3** shows moderate script awareness, similar to LaBSE
- **Vyakyarth's low consistency** (0.343) explains its dramatic Devanagari improvement: it learned different representations for each script
- **LaBSE** balances script awareness with cross-script retrieval capability

### 5.6 VBT Corpus: Cross-Lingual Retrieval

To evaluate cross-lingual performance on a substantive corpus, we benchmarked the models on 168 verses from the Vijñānabhairavatantra (VBT) with their English translations.

#### Cross-Lingual Retrieval (English → Sanskrit)

| Metric | Vyakyarth | LaBSE | E5-multilingual | MuRIL | BGE-M3 |
|--------|-----------|-------|-----------------|-------|--------|
| MRR (En→Sa) | 0.162 | 0.257 | **0.317** | 0.074 | 0.278 |
| Recall@1 | 0.097 | 0.161 | **0.226** | 0.000 | 0.161 |
| Recall@3 | 0.097 | 0.258 | 0.323 | 0.065 | **0.355** |

#### Translation→Verse Matching

This measures how well models match English translations back to their source Sanskrit verses — useful for cross-referencing commentarial traditions.

| Model | Translation→Verse MRR |
|-----------------|-----------------------|
| E5-multilingual | **0.820** |
| BGE-M3 | 0.766 |
| LaBSE | 0.575 |
| Vyakyarth | 0.457 |
| MuRIL | 0.229 |

#### VBT Similarity Discrimination (59 similarity + 39 dissimilarity pairs)

| Model | Similar | Dissimilar | Discrimination |
|-----------------|---------|------------|----------------|
| LaBSE | 0.488 | 0.266 | **0.222** |
| Vyakyarth | 0.484 | 0.277 | **0.207** |
| BGE-M3 | 0.603 | 0.462 | 0.140 |
| E5-multilingual | 0.905 | 0.844 | 0.060 |
| MuRIL | 0.995 | 0.992 | 0.003 |

**Key findings**:

1. **E5-multilingual dominates cross-lingual retrieval** — 0.820 Translation→Verse MRR is substantially higher than other models
2. **MuRIL fails completely** — 0.0 Recall@1 means it never ranks the correct Sanskrit verse first when given an English translation
3. **LaBSE has best discrimination** — 0.222 gap between similar and dissimilar pairs, slightly better than Vyakyarth (0.207)
4. **LaBSE is 2.4× faster than Vyakyarth** — 4.30ms vs 10.48ms per text encoding
5. **E5's high retrieval contradicts its poor discrimination** — it retrieves well but can't threshold reliably (everything clusters at 0.84–0.91)

### 5.7 Semantic Search Implications

The discrimination metric is critical for semantic search: you need models that assign high similarity to genuinely related passages and low similarity to unrelated ones.

#### The Core Problem

| Model | VBT Similar | VBT Dissimilar | Gap | Practical Impact |
|-----------------|-------------|----------------|-------|------------------|
| LaBSE | 0.488 | 0.266 | **0.222** | Best ranking |
| Vyakyarth | 0.484 | 0.277 | **0.207** | Good ranking |
| BGE-M3 | 0.603 | 0.462 | 0.140 | Marginal ranking |
| E5-multilingual | 0.905 | 0.844 | 0.060 | Unreliable ranking |
| MuRIL | 0.995 | 0.992 | 0.003 | **Useless** |

**MuRIL** is essentially useless for semantic search — it assigns 99.5% similarity to a verse and its translation, but also 99.2% similarity to completely unrelated verses. Search results would be noise.

**E5-multilingual** has the same problem at smaller scale: everything clusters in the 0.84–0.91 range. When searching for dhāraṇā techniques, you'd retrieve meditation verses mixed with cosmological verses indiscriminately.

**LaBSE** discriminates best — related texts score ~0.49, unrelated texts ~0.27. Combined with its fast encoding speed (4.30ms/text), it's the recommended choice for most Sanskrit retrieval tasks.

#### Model Selection by Goal

| Goal | Best Model | Rationale |
|---------------------------------|------------------------|-----------|
| **General Sanskrit retrieval** | **LaBSE** | Best discrimination (0.222) + fast (4.30ms) |
| Cross-lingual retrieval (En↔Sa) | E5-multilingual | Highest Translation→Verse MRR (0.820) |
| Script-agnostic matching | E5-multilingual | 0.901 transliteration consistency with usable discrimination |
| Maximum recall | BGE-M3 | Perfect Recall@3 (1.0) on IAST |

#### Recommended Approach

**For most Sanskrit work, use LaBSE alone.** It offers:
- Best semantic discrimination (0.222 on VBT corpus)
- Fast encoding (4.30ms/text — 2.4× faster than Vyakyarth)
- Good IAST retrieval (0.759 MRR)
- Peer-reviewed methodology (ACL 2022)

**For cross-lingual work (English ↔ Sanskrit)**, consider a two-stage pipeline:
1. **Coarse retrieval** with E5-multilingual (0.820 Translation→Verse MRR)
2. **Reranking** with LaBSE (better discrimination for final ranking)

The irony: the Sanskrit-specific Vyakyarth model underperforms LaBSE on both discrimination (0.207 vs 0.222) and speed (10.48ms vs 4.30ms). General multilingual models trained on massive corpora outperform specialized Indic models.

### 5.8 Performance Characteristics

| Model | Load Time | Encode Time (CPU) | Embedding Dim |
|-------|-----------|-------------------|---------------|
| LaBSE | 4.83s | **4.30ms/text** | 768 |
| MuRIL | 4.23s | 7.78ms/text | 768 |
| Vyakyarth | 1.81s | 10.48ms/text | 768 |
| E5-multilingual | 4.19s | 12.99ms/text | 1024 |
| BGE-M3 | 4.31s | 14.14ms/text | 1024 |

*Benchmarked on CPU (Intel Xeon)*

**Notes**:
- **LaBSE is fastest with usable quality** (4.30ms/text) — best speed/quality tradeoff
- **MuRIL is fast** (7.78ms/text) but its poor discrimination (0.003) makes speed irrelevant
- **Vyakyarth loads fastest** (1.81s) but encodes 2.4× slower than LaBSE
- **1024-dim models (E5, BGE-M3) are slower** due to larger embeddings

### 5.9 Benchmark Conclusions

1. **Use LaBSE for most Sanskrit work** — best discrimination (0.222), fast encoding (4.30ms), good retrieval (0.759 MRR)
2. **Always transliterate IAST to Devanagari** before embedding — all models improve substantially (17–42% MRR gain)
3. **Use E5-multilingual for cross-lingual retrieval** — 0.820 Translation→Verse MRR on VBT corpus
4. **Avoid MuRIL for semantic search** — despite academic citations, its mean-pooled embeddings have near-zero discrimination (0.003), making it useless for retrieval
5. **BGE-M3 for maximum recall** — achieves perfect Recall@3 on IAST (1.0) but mediocre precision
6. **Vyakyarth underperforms LaBSE** — slower (10.48ms vs 4.30ms) with worse discrimination (0.207 vs 0.222)

**Model rankings by use case**:

| Use Case | Best | Second | Avoid |
|----------|------|--------|-------|
| **General Sanskrit retrieval** | **LaBSE** | Vyakyarth | MuRIL |
| Cross-lingual (En↔Sa) | E5-multilingual | BGE-M3 | MuRIL |
| Semantic discrimination | LaBSE | Vyakyarth | MuRIL, E5 |
| Maximum recall | BGE-M3 | E5-multilingual | Vyakyarth |
| IAST-only corpus | LaBSE | E5-multilingual | MuRIL |
| Encoding speed | LaBSE | MuRIL* | BGE-M3 |

*MuRIL is fast but useless for retrieval due to poor discrimination.

### 5.10 ByT5-Sanskrit Preprocessing Evaluation

We evaluated whether preprocessing Sanskrit text with ByT5-Sanskrit (segmentation and/or lemmatization) improves embedding quality for retrieval tasks.

**Preprocessing modes tested**:
- **Segmentation**: Sandhi splitting (e.g., "ūrdhveprāṇo" → "ūrdhve prāṇo")
- **Lemmatization**: Reduction to dictionary forms (e.g., "prāṇāpānau" → "prāṇa apāna")
- **Both**: Segmentation followed by lemmatization

#### Retrieval Impact (MRR)

| Preprocessing | Vyakyarth | LaBSE | E5-multilingual |
|---------------|-----------|-------|-----------------|
| Raw IAST | 0.509 | **0.759** | 0.714 |
| Segmented | 0.505 | 0.643 | 0.694 |
| Lemmatized | 0.485 | 0.465 | 0.705 |
| Seg + Lemma | 0.502 | 0.439 | 0.411 |

#### MRR Delta from Preprocessing (negative = worse)

| Model | Segmented | Lemmatized | Both |
|-------|-----------|------------|------|
| Vyakyarth | -0.004 | -0.024 | -0.006 |
| LaBSE | **-0.116** | **-0.293** | **-0.319** |
| E5-multilingual | -0.020 | -0.010 | **-0.304** |

**Key findings**:

1. **ByT5-Sanskrit preprocessing hurts retrieval performance** — All models showed MRR degradation with preprocessing, contrary to the intuition that cleaner text would improve similarity matching

2. **LaBSE most negatively affected** — Dropped from 0.759 to 0.439 MRR (-42%) with combined preprocessing. LaBSE appears to have learned effective representations from unsegmented Sanskrit

3. **Vyakyarth most resilient** — Minimal impact (-0.006 MRR), suggesting its Indic-specific training already handles sandhi internally

4. **Lemmatization worse than segmentation** — Reducing inflected forms to lemmas loses grammatical information that embeddings capture

5. **Combined preprocessing compounds errors** — E5-multilingual dropped from 0.714 to 0.411 (-42%) with both steps applied

#### Similarity Discrimination with Preprocessing

| Preprocessing | Vyakyarth | LaBSE | E5-multilingual |
|---------------|-----------|-------|-----------------|
| Raw IAST | 0.084 | **0.120** | 0.046 |
| Segmented | **0.154** | 0.041 | 0.034 |
| Lemmatized | 0.109 | 0.027 | 0.014 |

**Interesting finding**: Vyakyarth's discrimination *improved* with segmentation (0.084 → 0.154), even though retrieval MRR decreased slightly. This suggests segmentation may help Vyakyarth distinguish semantic categories while slightly hurting exact match retrieval. LaBSE and E5 both degraded on discrimination.

#### Why Preprocessing Hurts

Several hypotheses explain why ByT5-Sanskrit preprocessing degrades embedding quality:

1. **Information loss**: Lemmatization removes grammatical markers (case, number, tense) that carry semantic information
2. **Domain mismatch**: Embedding models were trained on naturally-occurring Sanskrit text, not linguistically normalized forms
3. **Tokenization disruption**: Segmented text may tokenize differently, fragmenting learned subword patterns
4. **Training data alignment**: LaBSE and E5 likely saw sandhi-fused forms in training; segmented forms appear as novel sequences

#### Recommendation

**Do not use ByT5-Sanskrit preprocessing for retrieval tasks.** Instead:
- Transliterate to Devanagari (17–42% MRR improvement)
- Use raw text without segmentation or lemmatization
- Reserve ByT5-Sanskrit for linguistic analysis tasks (morphological tagging, dependency parsing) where it achieves SOTA results

### 5.11 Recommended Transliteration Pipeline

```python
from aksharamukha import transliterate
from sentence_transformers import SentenceTransformer

def embed_sanskrit(text: str, model: SentenceTransformer, source_script: str = "IAST") -> np.ndarray:
    """
    Embed Sanskrit text with automatic Devanagari transliteration.

    Args:
        text: Sanskrit text in any script
        model: SentenceTransformer model
        source_script: Input script (IAST, Devanagari, ITRANS, HK, etc.)

    Returns:
        Embedding vector
    """
    # Normalize to Devanagari for best embedding quality
    if source_script != "Devanagari":
        text = transliterate.process(source_script, "Devanagari", text)

    return model.encode([text])[0]

# Usage
model = SentenceTransformer("sentence-transformers/LaBSE")
embedding = embed_sanskrit("ūrdhve prāṇo hy adho jīvo", model, source_script="IAST")
```

---

## 6. Comparative Summary

Based on both literature review and empirical benchmarking on 5 models (VBT corpus: 168 verses, 59 similarity pairs, 39 dissimilarity pairs):

| Use Case | Recommended Model | Script | Rationale |
|----------|-------------------|--------|-----------|
| **General Sanskrit retrieval** | **LaBSE** | Devanagari | Best discrimination (0.222) + fast (4.30ms) |
| **Cross-lingual (En↔Sa)** | E5-multilingual | Either | 0.820 Translation→Verse MRR on VBT corpus |
| **Maximum recall** | BGE-M3 | Either | Perfect Recall@3 on IAST (1.0), good Devanagari |
| **Devanagari internal retrieval** | E5-multilingual | Devanagari | Perfect MRR (1.0) on Devanagari corpus |
| **Academic publication** | LaBSE | Devanagari | Peer-reviewed (ACL 2022); MuRIL fails for retrieval |
| **IAST-only corpus** | LaBSE | IAST | 0.759 MRR on IAST (best of 5 models) |
| **Linguistic analysis** | ByT5-Sanskrit | Either | SOTA segmentation/lemmatization/parsing |
| **Domain-specific similarity** | Custom fastText | — | Trainable on small corpora |

**Critical recommendations**:
1. **Use LaBSE for most Sanskrit work** — best discrimination (0.222), fastest encoding (4.30ms), peer-reviewed.
2. **Transliterate IAST → Devanagari** before embedding. All models improve 12–42% on MRR with Devanagari input.
3. **For cross-lingual retrieval**, use E5-multilingual (0.820 MRR) with optional LaBSE reranking.
4. **Do NOT use MuRIL for semantic search** — despite academic citations, it has near-zero discrimination (0.003) and clusters all text at ~0.99 similarity.
5. **Do NOT preprocess with ByT5-Sanskrit for retrieval** — segmentation/lemmatization hurts MRR by up to 42% (see Section 5.10). Use ByT5-Sanskrit only for linguistic analysis tasks.

---

## 7. Recommended Architecture for Sanskrit Semantic Search

For a pgvector-based retrieval system over texts like the Vijñānabhairava:

```
┌─────────────────────────────────────────────────────────────┐
│                     INDEXING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Raw Sanskrit (IAST/Devanagari/regional scripts)            │
│        │                                                     │
│        ▼                                                     │
│  aksharamukha (normalize to Devanagari) ←── CRITICAL STEP   │
│        │                                                     │
│        ▼                                                     │
│  LaBSE Embedding (sentence-transformers/LaBSE)              │
│  • Best discrimination (0.222) + fastest (4.30ms/text)      │
│        │                                                     │
│        ▼                                                     │
│  pgvector (HNSW index, cosine distance)                     │
│                                                              │
│  NOTE: Do NOT use ByT5-Sanskrit preprocessing for retrieval │
│  (see Section 5.10 — segmentation/lemmatization hurts MRR)  │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Query (English or Sanskrit in any script)             │
│        │                                                     │
│        ▼                                                     │
│  Script Detection + aksharamukha → Devanagari               │
│        │                                                     │
│        ▼                                                     │
│  Query Expansion (LLM: technical terms, synonyms)           │
│  e.g., "breath practice" → प्राण, कुम्भक, धारणा            │
│        │                                                     │
│        ▼                                                     │
│  LaBSE Embedding                                             │
│        │                                                     │
│        ├──────────────┐                                      │
│        ▼              ▼                                      │
│  Dense retrieval   Sparse retrieval (BM25)                  │
│  (pgvector)        (Devanagari technical vocabulary)        │
│        │              │                                      │
│        └──────┬───────┘                                      │
│               ▼                                              │
│  Reciprocal Rank Fusion                                     │
│               │                                              │
│               ▼                                              │
│  Final ranked results                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Rationale for architecture choices**:

1. **LaBSE as single embedding model**: Best discrimination (0.222), fastest encoding (4.30ms), good retrieval (0.759 MRR). No need for two-stage pipeline in most cases.
2. **Devanagari normalization**: Empirical benchmarks show 17–42% MRR improvement over IAST for all models tested
3. **No ByT5-Sanskrit preprocessing**: Our benchmarks show segmentation/lemmatization *hurts* retrieval (up to -42% MRR). Embedding models have learned effective representations from naturally-occurring Sanskrit text
4. **Hybrid retrieval**: Dense embeddings capture conceptual similarity but struggle with technical vocabulary (द्वादशान्त, कुम्भक, भैरव). Sparse retrieval handles exact terminology. Fusion combines strengths

**For cross-lingual retrieval (English ↔ Sanskrit)**, consider adding E5-multilingual (0.820 Translation→Verse MRR) as a first-stage retriever with LaBSE reranking.

---

## 8. Open Questions and Research Gaps

1. **No Sanskrit sentence-transformer**: The field lacks a model specifically trained for Sanskrit sentence similarity with proper evaluation. Fine-tuning MuRIL or XLM-R on Sanskrit paraphrase/NLI data would be valuable.

2. **Benchmark datasets**: No standard Sanskrit semantic similarity benchmark exists. Creating one from parallel translations (e.g., Vijñānabhairava commentarial tradition) would enable rigorous model comparison. *(The benchmark corpus used in Section 5 is a starting point but covers only tantric/yogic vocabulary.)*

3. **Script normalization**: Our benchmarks confirm Devanagari significantly outperforms IAST, but systematic study across regional scripts (Śāradā, Grantha, Telugu, etc.) is still lacking.

4. **Classical vs. modern Sanskrit**: Pre-trained models favor modern Sanskrit (Wikipedia). Our tantric text corpus showed all models can achieve good retrieval with Devanagari — performance on other classical registers (kāvya, śāstra, etc.) needs investigation.

5. **Evaluation on retrieval tasks**: Most Indic benchmarks focus on classification/NLI. Retrieval-specific evaluation (e.g., finding parallel passages across manuscripts) needs development.

6. **Sandhi-aware embeddings**: Current approaches require pre-segmentation. End-to-end models that handle unsegmented Sanskrit would be valuable.

7. **Why does Devanagari outperform IAST?**: Our benchmarks show consistent 17–42% MRR improvement but the mechanism is unclear. Hypotheses include: (a) more Devanagari in training data, (b) better tokenization of Devanagari, (c) IAST diacritics fragmenting into unusual subwords. This warrants investigation.

8. **E5's tight clustering problem**: E5-multilingual achieves high retrieval scores but poor discrimination (0.017–0.046). Understanding why general multilingual models cluster Sanskrit text so tightly could inform future model development.

9. **Why does morphological preprocessing hurt retrieval?**: Our ByT5-Sanskrit experiments show that segmentation and lemmatization *degrade* retrieval quality (up to -42% MRR), contrary to the intuition that cleaner text should help. Possible explanations include: (a) embedding models learned from naturally-occurring sandhi-fused text, (b) lemmatization loses grammatical information encoded in embeddings, (c) segmented text tokenizes differently, disrupting learned patterns. This counterintuitive result deserves deeper investigation — when does linguistic normalization help vs. hurt neural models?

10. **Model-specific preprocessing effects**: Vyakyarth's discrimination *improved* with segmentation (0.084 → 0.154) while retrieval MRR stayed flat. Understanding which models benefit from which preprocessing could enable model-specific pipelines.

---

## 9. Practical Recommendations

### For immediate deployment:
1. **Use LaBSE** (`sentence-transformers/LaBSE`) — best discrimination (0.222), fastest encoding (4.30ms)
2. **Always transliterate to Devanagari** using `aksharamukha` before embedding — 17–42% MRR improvement
3. Implement hybrid dense+sparse retrieval with Devanagari BM25 index
4. Use LLM-based query expansion for English→Sanskrit bridging (expand to Devanagari terms)
5. **Do NOT preprocess with ByT5-Sanskrit** — segmentation/lemmatization hurts retrieval (see Section 5.10)

### For cross-lingual retrieval (English ↔ Sanskrit):
1. Use `intfloat/multilingual-e5-large` — 0.820 Translation→Verse MRR on VBT corpus
2. Consider LaBSE reranking for better discrimination
3. Best for matching English translations back to Sanskrit source verses

### For maximum recall (at cost of precision):
1. Use `BAAI/bge-m3` with Devanagari input — perfect Recall@3 (1.0)
2. Use LaBSE reranking to filter false positives

### For academic rigor:
1. **Use LaBSE** (peer-reviewed ACL 2022) — NOT MuRIL, which has near-zero discrimination (0.003)
2. Report preprocessing steps (script normalization only — not segmentation/lemmatization for retrieval)
3. Create and release evaluation dataset for reproducibility
4. Report results on both IAST and Devanagari to enable comparison

### For best quality:
1. Normalize all text to Devanagari via aksharamukha
2. Use LaBSE for embedding (raw text, no morphological preprocessing)
3. Fine-tune on domain-specific Sanskrit pairs if available
4. Train custom fastText on your corpus as sparse retrieval component

### When to use ByT5-Sanskrit:
ByT5-Sanskrit achieves SOTA on linguistic analysis tasks but *hurts* retrieval quality. Use it for:
- Morphological tagging and dependency parsing
- Sandhi analysis for linguistic research
- OCR post-correction
- Preprocessing for machine translation pipelines
- **NOT** for semantic search or retrieval systems

---

## 10. Model Quick Reference

### 10.1 Recommended: LaBSE with Devanagari Transliteration

```python
from sentence_transformers import SentenceTransformer
from aksharamukha import transliterate
import numpy as np

# Load LaBSE — best discrimination (0.222) + fastest (4.30ms/text)
model = SentenceTransformer("sentence-transformers/LaBSE")

def embed_sanskrit(text: str, source_script: str = "IAST"):
    """Embed Sanskrit with automatic Devanagari normalization."""
    if source_script != "Devanagari":
        text = transliterate.process(source_script, "Devanagari", text)
    return model.encode([text])[0]

def search_sanskrit(query: str, corpus_embeddings: np.ndarray, corpus_texts: list,
                    top_k: int = 10, source_script: str = "IAST"):
    """Search Sanskrit corpus with LaBSE embeddings."""
    query_emb = embed_sanskrit(query, source_script)
    scores = np.dot(corpus_embeddings, query_emb)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(corpus_texts[i], scores[i]) for i in top_indices]

# Usage
embedding = embed_sanskrit("ūrdhve prāṇo hy adho jīvo", source_script="IAST")
# Equivalent to embedding "ऊर्ध्वे प्राणो ह्यधो जीवो"
```

### 10.2 Alternative Models

```python
from sentence_transformers import SentenceTransformer

# For cross-lingual retrieval (English ↔ Sanskrit)
model = SentenceTransformer("intfloat/multilingual-e5-large")  # 0.820 Translation→Verse MRR

# For maximum recall
model = SentenceTransformer("BAAI/bge-m3")  # Perfect Recall@3 (1.0)

# Option 2: Indic-optimized (requires Devanagari input)
model = SentenceTransformer("krutrim-ai-labs/Vyakyarth")

# Always use Devanagari for best results
embeddings = model.encode(["ऊर्ध्वे प्राणो ह्यधो जीवो"])
```

### 10.3 MuRIL with Manual Pooling

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
model = AutoModel.from_pretrained("google/muril-base-cased")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

text = "ऊर्ध्वे प्राणो ह्यधो जीवो"  # Devanagari preferred for MuRIL
encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    output = model(**encoded)
embedding = mean_pooling(output, encoded["attention_mask"])
```

### 10.4 FastText for Domain-Specific Search

```python
import fasttext
import fasttext.util
import numpy as np

# Download pre-trained (modern Sanskrit, ~1.2GB)
fasttext.util.download_model('sa', if_exists='ignore')
ft = fasttext.load_model('cc.sa.300.bin')

# Or train custom on your corpus:
# ft = fasttext.train_unsupervised('your_corpus.txt', model='skipgram', dim=300)

def sentence_vec(text, model):
    words = text.split()  # Assumes pre-segmented
    vecs = [model.get_word_vector(w) for w in words if w]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.get_dimension())
```

### 10.5 ByT5-Sanskrit for Linguistic Analysis (NOT for Retrieval)

⚠️ **Warning**: Do not use ByT5-Sanskrit preprocessing for semantic search or retrieval. Our benchmarks show it degrades MRR by up to 42% (see Section 5.10). Use only for linguistic analysis tasks.

```python
# For linguistic analysis tasks (morphological tagging, parsing, OCR correction)
# NOT for retrieval preprocessing
# See: https://github.com/sebastian-nehrdich/byt5-sanskrit-analyzers

from transformers import T5ForConditionalGeneration, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("chronbmm/sanskrit5-multitask")
model = T5ForConditionalGeneration.from_pretrained("chronbmm/sanskrit5-multitask")

def segment_sanskrit(text):
    """Sandhi splitting - use for linguistic analysis, NOT retrieval."""
    inputs = tokenizer(f"S{text}", return_tensors="pt")  # S = segmentation task
    outputs = model.generate(**inputs, max_length=512, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def lemmatize_sanskrit(text):
    """Lemmatization - use for linguistic analysis, NOT retrieval."""
    inputs = tokenizer(f"L{text}", return_tensors="pt")  # L = lemmatization task
    outputs = model.generate(**inputs, max_length=512, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example: linguistic analysis
text = "ūrdhve prāṇo hy adho jīvo"
print(f"Segmented: {segment_sanskrit(text)}")
print(f"Lemmatized: {lemmatize_sanskrit(text)}")

# For retrieval, use raw Devanagari text instead:
# embedding = labse_model.encode(["ऊर्ध्वे प्राणो ह्यधो जीवो"])
```

---

## 11. References

### Peer-Reviewed

- Khanuja et al. (2021). "MuRIL: Multilingual Representations for Indian Languages." arXiv:2103.10730
- Nehrdich, Hellwig, Keutzer (2024). "One Model is All You Need: ByT5-Sanskrit, a Unified Model for Sanskrit NLP Tasks." EMNLP 2024 Findings. [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.805/)
- Lugli et al. (2022). "Embeddings Models for Buddhist Sanskrit." LREC 2022.
- Feng et al. (2022). "Language-agnostic BERT Sentence Embedding (LaBSE)." ACL 2022.
- Bojanowski et al. (2017). "Enriching Word Vectors with Subword Information." TACL. (FastText)

### Technical Reports / Preprints

- Krutrim Team (2024). "Krutrim LLM: A Novel Tokenization Strategy for Multilingual Indic Languages." arXiv:2407.12481
- Krutrim Team (2025). "Krutrim LLM: Multilingual Foundational Model." arXiv:2502.09642
- AI4Bharat IndicNLP Catalog: https://ai4bharat.github.io/indicnlp_catalog/

### Tools

- **Aksharamukha**: Script transliteration library supporting 100+ scripts. PyPI: `aksharamukha==2.3`. [Documentation](https://aksharamukha.appspot.com/). Critical for IAST→Devanagari normalization (see Section 5).

### Model Repositories

| Model | HuggingFace ID | Sentence-Native | Notes |
|-------|----------------|-----------------|-------|
| **LaBSE** | `sentence-transformers/LaBSE` | ✅ | **⭐ RECOMMENDED** — Best discrimination (0.222) + fastest (4.30ms) |
| E5-multilingual | `intfloat/multilingual-e5-large` | ✅ | Best cross-lingual (0.820 MRR) — for En↔Sa retrieval |
| Vyakyarth | `krutrim-ai-labs/Vyakyarth` | ✅ | Good discrimination (0.207) but 2.4× slower than LaBSE |
| BGE-M3 | `BAAI/bge-m3` | ✅ | Best recall (1.0 R@3), mediocre discrimination (0.140) |
| MuRIL | `google/muril-base-cased` | ❌ (needs pooling) | ⚠️ **Unsuitable** — near-zero discrimination (0.003) |
| ByT5-Sanskrit | `chronbmm/sanskrit5-multitask` | ❌ (task model) | Linguistic analysis only; hurts retrieval |
| FastText Sanskrit | `cc.sa.300.bin` via `fasttext.util.download_model('sa')` | ❌ (word-level) | Domain-specific training |

---

*Last updated: February 2025 (VBT corpus benchmark with 168 verses, 59 similarity + 39 dissimilarity pairs)*

*This survey was prepared for practical application to Sanskrit manuscript retrieval systems. Section 5 contains original empirical benchmarks conducted on tantric/yogic Sanskrit texts, including the VBT corpus (168 Vijñānabhairavatantra verses with English translations) for cross-lingual evaluation, and assessment of ByT5-Sanskrit preprocessing effects. The benchmark code is available in this repository (`benchmark_embeddings.py`). Corrections and additions welcome.*
