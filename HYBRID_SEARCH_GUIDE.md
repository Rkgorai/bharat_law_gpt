# Hybrid Search Implementation Guide

## What Changed?

Your Bharat Law GPT now uses **Hybrid Search** combining:
- **BM25**: Keyword-based search (like traditional search engines)
- **Vector Search**: Semantic similarity using embeddings

This provides better retrieval by combining the strengths of both methods!

## Benefits

1. **Better keyword matching**: BM25 finds exact terms and legal keywords
2. **Semantic understanding**: Vector search finds conceptually similar content
3. **Improved accuracy**: Weighted combination of both methods
4. **Legal-specific**: Great for finding specific acts/sections while understanding context

## Files Modified

1. **requirements.txt**: Added `rank-bm25` package
2. **src/hybrid_vectorstore.py**: NEW - Hybrid search implementation
3. **src/search.py**: Updated to use HybridVectorStore
4. **build_db.py**: Updated to build hybrid indexes

## How to Use

### 1. Install New Dependencies

```bash
pip install -r requirements.txt
```

### 2. Rebuild Your Database

**IMPORTANT**: You must rebuild the database to create both BM25 and vector indexes:

```bash
python build_db.py
```

This will:
- Delete the old FAISS-only database
- Create new hybrid indexes (BM25 + Vector)
- Process all your legal documents

### 3. Run Your Application

```bash
streamlit run app_ui.py
```

The app will automatically use hybrid search now!

## Configuration

You can adjust the weights in `src/search.py` and `build_db.py`:

```python
HybridVectorStore(
    persist_dir=DB_PATH,
    bm25_weight=0.5,    # Weight for keyword search (0-1)
    vector_weight=0.5   # Weight for semantic search (0-1)
)
```

**Recommended settings:**
- **Equal balance (0.5, 0.5)**: Good default for most cases
- **More BM25 (0.7, 0.3)**: When exact legal terms are critical
- **More Vector (0.3, 0.7)**: When conceptual similarity matters more

## Technical Details

### BM25 Search
- Uses Okapi BM25 algorithm
- Tokenizes text by splitting on whitespace and lowercasing
- Good for finding specific legal sections, acts, and keywords

### Vector Search
- Uses FAISS with L2 distance
- Sentence-Transformers for embeddings
- Captures semantic meaning and context

### Score Combination
1. Retrieve top candidates from both methods
2. Normalize scores to 0-1 range
3. Combine using weighted sum
4. Return top K results by combined score

## Troubleshooting

**Error: "No module named 'rank_bm25'"**
- Run: `pip install rank-bm25`

**Old database format error**
- Delete `db/faiss_store` folder and rebuild: `python build_db.py`

**Poor search results**
- Try adjusting weights in the configuration
- Ensure your documents are properly formatted
- Check that PDFs are in `legal_docs/` folder

## Performance

- **Build time**: Slightly longer (adds BM25 indexing)
- **Query time**: ~Same speed (searches run in parallel)
- **Storage**: Minimal increase (BM25 index is small)
- **Accuracy**: Significantly improved for legal documents

## Next Steps

1. Rebuild your database with `python build_db.py`
2. Test queries that combine keywords and concepts
3. Adjust weights if needed based on your specific use case
4. Monitor search quality and iterate

## Example Queries That Benefit

- "What is the punishment for theft?" → BM25 finds "theft", Vector understands "punishment"
- "Rights of accused" → BM25 matches exact terms, Vector finds related rights
- "Section 302 IPC" → BM25 excels at finding specific section numbers
- "Consumer protection laws" → Vector finds conceptually related consumer rights

Enjoy your improved search! 🚀
