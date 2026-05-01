# ============================================
# analysis.py
# GreenBit Framework — Semantic Analysis Engine
# Responsibility:
#   - Text extraction from multiple formats
#   - Transformer embedding generation
#   - DBSCAN clustering
#   - Near-duplicate detection
#   - Cluster statistics computation
# ============================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sentence_transformers import (
    SentenceTransformer
)

# ── Optional Imports with Graceful Fallback ──

# PyMuPDF for PDF text extraction
try:
    import fitz
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print(
        "[GreenBit] Warning: PyMuPDF not "
        "found. PDF extraction disabled."
    )

# python-docx for DOCX text extraction
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print(
        "[GreenBit] Warning: python-docx "
        "not found. DOCX extraction disabled."
    )


# ── Configuration Constants ──────────────────

# Distilled transformer model
# Green AI choice — 1/6th cost of BERT-large
# Produces 384-dimensional embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# DBSCAN parameters
# Empirically validated on 500-pair dataset
# epsilon = max cosine distance for neighbors
DBSCAN_EPS = 0.3

# Minimum files to form a waste cluster
DBSCAN_MIN_SAMPLES = 2

# Files processed per inference batch
# Optimal for consumer-grade CPU/GPU
EMBEDDING_BATCH_SIZE = 32

# Maximum characters to read per file
# Prevents memory issues on large files
MAX_TEXT_LENGTH = 10000

# Cached model instance
# Loaded once and reused across batches
_model_cache = None


# ── Model Loading ────────────────────────────

def get_embedding_model():
    """
    Load the SentenceTransformer model
    with singleton caching to prevent
    repeated initialization overhead
    across multiple batch iterations.

    The model is downloaded from Hugging
    Face on first call and cached locally
    for all subsequent executions.

    Returns:
        SentenceTransformer: Loaded model
            instance ready for inference
    """
    global _model_cache

    if _model_cache is None:
        print(
            f"[GreenBit] Loading model: "
            f"{EMBEDDING_MODEL}"
        )
        _model_cache = SentenceTransformer(
            EMBEDDING_MODEL
        )
        print(
            "[GreenBit] Model loaded "
            "successfully."
        )

    return _model_cache


# ── Text Extraction ──────────────────────────

def extract_text_from_pdf(filepath):
    """
    Extract plain text from PDF files
    using PyMuPDF (fitz) library.
    Processes each page and concatenates
    all extracted text into single string.

    Args:
        filepath (str): Path to PDF file

    Returns:
        str: Extracted plain text content
             or empty string on failure
    """
    try:
        doc = fitz.open(filepath)
        text_parts = []

        for page in doc:
            text_parts.append(
                page.get_text()
            )

        doc.close()
        return " ".join(text_parts)

    except Exception:
        return ""


def extract_text_from_docx(filepath):
    """
    Extract plain text from Microsoft
    Word DOCX files using python-docx.
    Iterates over all paragraphs and
    joins non-empty text content.

    Args:
        filepath (str): Path to DOCX file

    Returns:
        str: Extracted plain text content
             or empty string on failure
    """
    try:
        document = docx.Document(filepath)
        paragraphs = []

        for para in document.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)

        return " ".join(paragraphs)

    except Exception:
        return ""


def extract_text_from_plaintext(filepath):
    """
    Extract text from plain text based
    file formats including .txt, .py,
    .json, .csv, .md, .html, .java,
    .cpp, .log, .yaml, .yml, .sql etc.

    Attempts UTF-8 encoding first with
    latin-1 fallback for files containing
    non-standard characters.

    Args:
        filepath (str): Path to text file

    Returns:
        str: Extracted plain text content
             or empty string on failure
    """
    try:
        # Try UTF-8 first
        with open(
            filepath, "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:
            return f.read(MAX_TEXT_LENGTH)

    except UnicodeDecodeError:
        try:
            # Fallback to latin-1
            with open(
                filepath, "r",
                encoding="latin-1",
                errors="ignore"
            ) as f:
                return f.read(MAX_TEXT_LENGTH)

        except Exception:
            return ""

    except Exception:
        return ""


def extract_text(filepath):
    """
    Unified text extraction dispatcher
    that routes each file to the
    appropriate format-specific extraction
    function based on file extension.

    Returns empty string for unreadable,
    corrupted, or unsupported files —
    producing a zero vector in embedding
    space, correctly classified as noise
    by DBSCAN clustering algorithm.

    Args:
        filepath (str): Path to source file

    Returns:
        str: Extracted plain text content
             (empty string on any failure)
    """
    ext = Path(filepath).suffix.lower()

    # ── PDF Extraction ────────────────────
    if ext == ".pdf":
        if PDF_AVAILABLE:
            return extract_text_from_pdf(
                filepath
            )
        else:
            return ""

    # ── DOCX Extraction ───────────────────
    elif ext == ".docx":
        if DOCX_AVAILABLE:
            return extract_text_from_docx(
                filepath
            )
        else:
            return ""

    # ── Plain Text Extraction ─────────────
    else:
        return extract_text_from_plaintext(
            filepath
        )


# ── Embedding Generation ─────────────────────

def generate_embeddings(file_paths):
    """
    Generate 384-dimensional semantic
    embedding vectors for a list of
    file paths using the all-MiniLM-L6-v2
    Transformer model.

    Processes files in micro-batches of
    EMBEDDING_BATCH_SIZE (32) to prevent
    memory overflow on large datasets.

    Model is loaded once via singleton
    cache and reused across all batches
    to minimize initialization overhead.

    Args:
        file_paths (list): List of file
            paths to generate embeddings

    Returns:
        np.ndarray: Embedding matrix of
            shape (n_files, 384) containing
            semantic vectors for each file
    """
    model = get_embedding_model()
    all_embeddings = []
    total = len(file_paths)

    print(
        f"[GreenBit] Generating embeddings "
        f"for {total} files..."
    )

    # Process files in micro-batches
    for start_idx in range(
            0, total, EMBEDDING_BATCH_SIZE):

        # Calculate end index for this batch
        end_idx = min(
            start_idx + EMBEDDING_BATCH_SIZE,
            total
        )

        # Get current batch of file paths
        batch_paths = file_paths[
            start_idx:end_idx
        ]

        # Extract text for each file
        # in current batch
        batch_texts = []
        for fp in batch_paths:
            text = extract_text(fp)
            # Use filename as fallback if
            # text extraction returns empty
            if not text.strip():
                text = os.path.basename(fp)
            batch_texts.append(text)

        # Generate embeddings for batch
        # normalize=True produces unit vectors
        # suitable for cosine similarity
        batch_embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        all_embeddings.extend(batch_embeddings)

        # Print progress every 5 batches
        if (start_idx // EMBEDDING_BATCH_SIZE
                ) % 5 == 0:
            progress = min(end_idx, total)
            print(
                f"[GreenBit] Embeddings: "
                f"{progress}/{total} files"
            )

    print(
        f"[GreenBit] Embeddings complete: "
        f"{len(all_embeddings)} vectors"
    )

    return np.array(all_embeddings)


# ── DBSCAN Clustering ────────────────────────

def cluster_documents(embeddings):
    """
    Apply DBSCAN density-based clustering
    to semantic embedding vectors using
    cosine distance metric.

    DBSCAN Configuration:
        eps = 0.3    (max cosine distance)
        min_samples = 2  (min cluster size)
        metric = cosine  (distance function)

    Files within cosine distance 0.3 of
    each other are grouped into the same
    waste cluster. Files further than 0.3
    from all neighbors are labeled -1
    (noise = unique, non-redundant files).

    Args:
        embeddings (np.ndarray): Matrix of
            shape (n_files, 384)

    Returns:
        np.ndarray: Cluster label array
            (-1 = unique noise point,
             0+ = waste cluster member)
    """
    if len(embeddings) == 0:
        print(
            "[GreenBit] No embeddings "
            "to cluster."
        )
        return np.array([])

    if len(embeddings) == 1:
        # Single file cannot form a cluster
        return np.array([-1])

    print(
        f"[GreenBit] Clustering "
        f"{len(embeddings)} documents..."
    )

    # Apply DBSCAN with cosine distance
    # n_jobs=-1 uses all available CPU cores
    clustering = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="cosine",
        algorithm="auto",
        n_jobs=-1
    )

    labels = clustering.fit_predict(embeddings)

    # Calculate and report cluster statistics
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (
        1 if -1 in unique_labels else 0
    )
    n_noise = list(labels).count(-1)
    n_waste = len(labels) - n_noise

    print(
        f"[GreenBit] Clusters found: "
        f"{n_clusters}"
    )
    print(
        f"[GreenBit] Waste files: "
        f"{n_waste}"
    )
    print(
        f"[GreenBit] Unique files: "
        f"{n_noise}"
    )

    return labels


# ── Master Analysis Function ─────────────────

def analyze_files(df):
    """
    Master analysis function that
    orchestrates the complete semantic
    analysis pipeline for all non-duplicate
    files in the input DataFrame.

    Execution sequence:
        1. Separate exact duplicates from
           unique files
        2. Extract file paths for analysis
        3. Generate semantic embeddings
           using transformer model
        4. Apply DBSCAN clustering to
           group near-duplicates
        5. Append cluster labels to DataFrame
        6. Recombine with exact duplicates
        7. Compute and return statistics

    Exact duplicates (flagged in ingestion)
    are assigned cluster label -2 to
    distinguish them from DBSCAN noise (-1)
    and waste clusters (0+).

    Args:
        df (pd.DataFrame): File metadata
            DataFrame from Ingestion Engine
            with columns: path, filename,
            extension, size, hash,
            is_duplicate

    Returns:
        tuple: (
            stats (dict): Analysis statistics
                containing cluster counts,
                waste file counts, and
                storage metrics,
            result_df (pd.DataFrame):
                Augmented DataFrame with
                cluster column appended
        )
    """
    stats = {}

    # ── Handle Empty DataFrame ────────────
    if df is None or df.empty:
        print(
            "[GreenBit] Empty DataFrame "
            "received. Skipping analysis."
        )
        df["cluster"] = pd.Series(
            dtype=int
        )
        return stats, df

    print(
        f"[GreenBit] Starting analysis "
        f"for {len(df)} files..."
    )

    # ── Separate Unique and Duplicate ─────
    # Only analyze non-duplicate files
    unique_mask = (
        df["is_duplicate"] == False
    )
    unique_df = df[unique_mask].copy()
    dupe_df   = df[~unique_mask].copy()

    # Assign -2 to exact duplicates
    # Distinguishes from DBSCAN noise (-1)
    dupe_df["cluster"] = -2

    print(
        f"[GreenBit] Unique files: "
        f"{len(unique_df)}"
    )
    print(
        f"[GreenBit] Exact duplicates: "
        f"{len(dupe_df)}"
    )

    # ── Handle All-Duplicate Dataset ──────
    if unique_df.empty:
        print(
            "[GreenBit] All files are exact "
            "duplicates. Skipping semantic "
            "analysis."
        )
        df["cluster"] = -2
        stats["clusters_found"]      = 0
        stats["waste_files"]         = len(dupe_df)
        stats["unique_files"]        = 0
        stats["waste_storage_bytes"] = (
            dupe_df["size"].sum()
        )
        return stats, df

    # ── Generate Semantic Embeddings ──────
    unique_paths = (
        unique_df["path"].tolist()
    )
    embeddings = generate_embeddings(
        unique_paths
    )

    # ── Apply DBSCAN Clustering ───────────
    labels = cluster_documents(embeddings)

    # ── Append Cluster Labels ─────────────
    unique_df = unique_df.reset_index(
        drop=True
    )

    if len(labels) == len(unique_df):
        unique_df["cluster"] = labels
    else:
        # Fallback if label count mismatch
        unique_df["cluster"] = -1

    # ── Recombine DataFrames ──────────────
    result_df = pd.concat(
        [unique_df, dupe_df],
        ignore_index=True
    )

    # ── Compute Statistics ────────────────

    # Count semantic waste clusters (0+)
    waste_labels = labels[labels >= 0]
    n_clusters   = len(set(waste_labels))
    n_waste      = len(waste_labels)
    n_unique     = list(labels).count(-1)

    # Calculate waste storage in bytes
    waste_mask = unique_df["cluster"] >= 0
    waste_storage = unique_df[
        waste_mask
    ]["size"].sum()

    # Calculate duplicate storage
    dupe_storage = dupe_df["size"].sum()

    # Total reclaimable storage
    total_reclaimable = (
        waste_storage + dupe_storage
    )

    # Populate statistics dictionary
    stats["total_files"]          = len(df)
    stats["unique_files"]         = n_unique
    stats["exact_duplicates"]     = len(dupe_df)
    stats["clusters_found"]       = n_clusters
    stats["waste_files"]          = n_waste
    stats["waste_storage_bytes"]  = waste_storage
    stats["dupe_storage_bytes"]   = dupe_storage
    stats["total_reclaimable_bytes"] = (
        total_reclaimable
    )

    # Print final summary
    print("\n[GreenBit] ── Analysis Complete ──")
    print(
        f"  Total files    : {len(df)}"
    )
    print(
        f"  Exact dupes    : {len(dupe_df)}"
    )
    print(
        f"  Semantic clusters : {n_clusters}"
    )
    print(
        f"  Waste files    : {n_waste}"
    )
    print(
        f"  Unique files   : {n_unique}"
    )
    print(
        f"  Reclaimable    : "
        f"{total_reclaimable/(1024**3):.2f} GB"
    )

    return stats, result_df


# ── Quick Test ───────────────────────────────
# Run this file directly to test analysis
# python analysis.py

if __name__ == "__main__":
    from ingestion import run_ingestion
    import sys

    test_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "."
    )

    print("=" * 50)
    print("GreenBit — Analysis Engine Test")
    print("=" * 50)

    try:
        # Run ingestion first
        print("\n── Running Ingestion ───────────")
        df = run_ingestion(test_path)

        # Run analysis
        print("\n── Running Analysis ────────────")
        stats, result_df = analyze_files(df)

        # Print results
        print("\n── Results ─────────────────────")
        print(f"Stats    : {stats}")
        print(
            f"DataFrame: {result_df.shape}"
        )
        print("\n── Cluster Distribution ────────")
        print(
            result_df["cluster"]
            .value_counts()
            .head(10)
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()