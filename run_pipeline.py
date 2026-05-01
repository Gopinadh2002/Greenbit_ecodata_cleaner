# ============================================
# run_pipeline.py
# GreenBit Framework — Pipeline Orchestrator
# Responsibility:
#   - Sequential pipeline coordination
#   - CodeCarbon emission tracking
#   - Net Environmental Gain calculation
#   - Energy ROI computation
#   - Graceful degradation handling
#   - Cross-platform path normalization
# ============================================

import os
import time
import pandas as pd
from ingestion import run_ingestion
from analysis  import analyze_files

# ── CodeCarbon with Graceful Fallback ────────
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
    print(
        "[GreenBit] CodeCarbon available. "
        "Emission tracking enabled."
    )
except ImportError:
    CODECARBON_AVAILABLE = False
    print(
        "[GreenBit] CodeCarbon not found. "
        "Running without emission tracking."
    )


# ── Environmental Constants ──────────────────

# Global average carbon intensity
# (kg CO₂ per kWh of electricity)
CARBON_INTENSITY = 0.475

# Annual energy consumption per GB of
# active data center storage
# Source: IEA Data Centers Report 2024
KWH_PER_GB_PER_YEAR = 0.000002

# Byte conversion constants
BYTES_TO_GB = 1 / (1024 ** 3)
BYTES_TO_MB = 1 / (1024 ** 2)
BYTES_TO_KB = 1 / 1024


# ── Environmental Calculations ───────────────

def calculate_net_environmental_gain(stats):
    """
    Calculate the Net Environmental Gain —
    the core Green AI verification metric
    that proves GreenBit delivers a positive
    ecological outcome.

    Formula:
        Annual Energy Saved =
            Reclaimable Storage (GB)
            × KWH_PER_GB_PER_YEAR

        Audit Energy Cost =
            CO₂ Emitted (kg)
            / Carbon Intensity (kg/kWh)

        Net Environmental Gain =
            Annual Energy Saved
            - Audit Energy Cost

        Energy ROI =
            Annual Energy Saved
            / Audit Energy Cost

    A Net Gain > 0 confirms that GreenBit
    saves more energy than it consumes.
    An Energy ROI > 1.0 confirms that the
    audit is environmentally justified.

    Args:
        stats (dict): Pipeline statistics
            containing co2_emissions and
            total_reclaimable_bytes values

    Returns:
        dict: Stats updated with:
            - audit_energy_kwh
            - annual_energy_saved_kwh
            - net_environmental_gain_kwh
            - energy_roi
            - co2_offset_kg
    """

    # ── Get CO₂ emissions from tracker ───
    co2_kg = stats.get(
        "co2_emissions", 0
    ) or 0.0

    # ── Calculate audit energy cost ───────
    # Convert CO₂ back to kWh consumed
    audit_energy_kwh = (
        co2_kg / CARBON_INTENSITY
        if CARBON_INTENSITY > 0
        else 0.0
    )

    # ── Calculate reclaimable storage ─────
    reclaimable_bytes = stats.get(
        "total_reclaimable_bytes", 0
    ) or 0
    reclaimable_gb = (
        reclaimable_bytes * BYTES_TO_GB
    )

    # ── Calculate annual energy savings ───
    # Energy saved by deleting redundant
    # storage from active server
    annual_energy_saved_kwh = (
        reclaimable_gb * KWH_PER_GB_PER_YEAR
    )

    # ── Calculate Net Environmental Gain ──
    net_gain_kwh = (
        annual_energy_saved_kwh
        - audit_energy_kwh
    )

    # ── Calculate Energy ROI ──────────────
    # Ratio of savings to audit cost
    # > 1.0 = environmentally beneficial
    energy_roi = (
        annual_energy_saved_kwh
        / audit_energy_kwh
        if audit_energy_kwh > 0
        else 0.0
    )

    # ── Calculate CO₂ offset ─────────────
    # CO₂ saved by removing redundant storage
    co2_offset_kg = (
        annual_energy_saved_kwh
        * CARBON_INTENSITY
    )

    # ── Update statistics dictionary ──────
    stats["audit_energy_kwh"] = (
        round(audit_energy_kwh, 6)
    )
    stats["annual_energy_saved_kwh"] = (
        round(annual_energy_saved_kwh, 6)
    )
    stats["net_environmental_gain_kwh"] = (
        round(net_gain_kwh, 6)
    )
    stats["energy_roi"] = (
        round(energy_roi, 2)
    )
    stats["co2_offset_kg"] = (
        round(co2_offset_kg, 6)
    )
    stats["reclaimable_gb"] = (
        round(reclaimable_gb, 3)
    )

    return stats


def calculate_storage_metrics(df, stats):
    """
    Calculate comprehensive storage metrics
    from the augmented DataFrame for display
    on the Streamlit dashboard KPI cards.

    Computes total storage, average file
    size, storage by file type, and size
    category distributions.

    Args:
        df (pd.DataFrame): Augmented file
            metadata DataFrame with cluster
            column from analysis engine
        stats (dict): Existing statistics
            dictionary from analysis engine

    Returns:
        dict: Stats enriched with storage
            metrics for dashboard rendering
    """

    if df is None or df.empty:
        return stats

    # ── Total storage metrics ─────────────
    total_bytes = df["size"].sum()
    total_gb    = total_bytes * BYTES_TO_GB
    avg_bytes   = df["size"].mean()
    avg_mb      = avg_bytes * BYTES_TO_MB

    stats["total_storage_bytes"] = total_bytes
    stats["total_storage_gb"]    = round(
        total_gb, 3
    )
    stats["avg_file_size_mb"]    = round(
        avg_mb, 3
    )
    stats["total_files"]         = len(df)

    # ── Exact duplicate metrics ───────────
    exact_dupes = df[
        df["is_duplicate"] == True
    ]
    stats["exact_duplicate_count"] = len(
        exact_dupes
    )
    stats["exact_duplicate_storage_gb"] = round(
        exact_dupes["size"].sum() * BYTES_TO_GB,
        3
    )

    # ── Semantic cluster metrics ──────────
    if "cluster" in df.columns:
        waste_files = df[df["cluster"] >= 0]
        stats["semantic_waste_count"] = len(
            waste_files
        )
        stats["semantic_waste_storage_gb"] = round(
            waste_files["size"].sum()
            * BYTES_TO_GB,
            3
        )
        stats["clusters_found"] = len(
            df[df["cluster"] >= 0]
            ["cluster"].unique()
        )

    # ── Unique files metrics ──────────────
    unique_files = df[
        (df["is_duplicate"] == False)
        & (df.get("cluster", -1) == -1)
    ] if "cluster" in df.columns else df[
        df["is_duplicate"] == False
    ]
    stats["unique_file_count"] = len(
        unique_files
    )

    return stats


# ── Pipeline Execution ───────────────────────

def stream_pipeline(folder):
    """
    Master pipeline execution function
    that orchestrates the complete GreenBit
    analysis pipeline from directory
    scanning through carbon audit completion.

    Execution sequence:
        1. Validate and normalize path
        2. Initialize CodeCarbon tracker
        3. Run Ingestion Engine
        4. Run Semantic Analysis Engine
        5. Stop emission tracker
        6. Calculate storage metrics
        7. Calculate Net Environmental Gain
        8. Return results to dashboard

    Implements graceful degradation for
    environments where CodeCarbon is
    unavailable — pipeline continues
    without emission tracking rather than
    failing entirely.

    Args:
        folder (str): Target directory path

    Returns:
        tuple: (
            result_df (pd.DataFrame):
                Augmented file metadata
                DataFrame with cluster
                labels appended,
            stats (dict):
                Complete statistics dict
                containing all KPI values,
                carbon metrics, and
                environmental gain data
        )

    Raises:
        ValueError: If directory not found
        Exception: For unexpected errors
    """

    # ── Step 1: Validate and normalize ────
    # Normalize path for cross-platform
    # compatibility (Windows/macOS/Linux)
    folder = os.path.normpath(folder)

    if not os.path.isdir(folder):
        raise ValueError(
            f"[GreenBit] Directory not "
            f"found: {folder}"
        )

    print("\n" + "=" * 50)
    print("GreenBit — Pipeline Starting")
    print("=" * 50)
    print(f"Target folder: {folder}")

    # Record pipeline start time
    pipeline_start = time.time()

    tracker  = None
    stats    = {}
    result_df = pd.DataFrame()

    # ── Step 2: Initialize Tracker ────────
    if CODECARBON_AVAILABLE:
        try:
            tracker = EmissionsTracker(
                # Suppress verbose logging
                log_level="error",
                # Measure every 10 seconds
                measure_power_secs=10,
                # Save emission log here
                output_dir="logs",
                # Use offline mode
                # (no API calls required)
                save_to_file=True,
                save_to_api=False
            )
            tracker.start()
            print(
                "[GreenBit] Emission tracking "
                "started."
            )

        except Exception as e:
            print(
                f"[GreenBit] Tracker init "
                f"failed: {e}. Continuing "
                f"without tracking."
            )
            tracker = None

    try:
        # ── Step 3: Run Ingestion ─────────
        print(
            "\n[GreenBit] Phase 1: "
            "Data Ingestion..."
        )
        ingestion_start = time.time()

        df = run_ingestion(folder)

        ingestion_time = (
            time.time() - ingestion_start
        )
        print(
            f"[GreenBit] Ingestion complete "
            f"in {ingestion_time:.1f}s"
        )

        # Handle empty directory
        if df.empty:
            print(
                "[GreenBit] No supported "
                "files found in directory."
            )
            stats["error"] = "empty_directory"
            stats["total_files"] = 0

            # Stop tracker if running
            if tracker:
                try:
                    emissions = tracker.stop()
                    stats["co2_emissions"] = (
                        float(emissions)
                        if emissions
                        else 0.0
                    )
                except Exception:
                    stats["co2_emissions"] = 0.0

            return df, stats

        # ── Step 4: Run Analysis ──────────
        print(
            "\n[GreenBit] Phase 2: "
            "Semantic Analysis..."
        )
        analysis_start = time.time()

        analysis_stats, result_df = (
            analyze_files(df)
        )

        analysis_time = (
            time.time() - analysis_start
        )
        print(
            f"[GreenBit] Analysis complete "
            f"in {analysis_time:.1f}s"
        )

        # Merge analysis stats into main stats
        stats.update(analysis_stats)

    except Exception as e:
        print(
            f"[GreenBit] Pipeline error: {e}"
        )
        # Stop tracker before re-raising
        if tracker:
            try:
                tracker.stop()
            except Exception:
                pass
        raise

    # ── Step 5: Stop Emission Tracker ─────
    print(
        "\n[GreenBit] Phase 3: "
        "Carbon Audit..."
    )

    if tracker is not None:
        try:
            emissions = tracker.stop()
            stats["co2_emissions"] = (
                float(emissions)
                if emissions and emissions > 0
                else 0.0
            )
            print(
                f"[GreenBit] CO₂ emitted: "
                f"{stats['co2_emissions']:.6f} kg"
            )

        except Exception as e:
            print(
                f"[GreenBit] Tracker stop "
                f"error: {e}"
            )
            stats["co2_emissions"] = 0.0
    else:
        # No tracker available
        stats["co2_emissions"] = None
        print(
            "[GreenBit] No emission data "
            "available."
        )

    # ── Step 6: Calculate Storage Metrics ─
    stats = calculate_storage_metrics(
        result_df, stats
    )

    # ── Step 7: Calculate Net Gain ────────
    stats = calculate_net_environmental_gain(
        stats
    )

    # ── Step 8: Record total pipeline time ─
    total_time = time.time() - pipeline_start
    stats["pipeline_time_seconds"] = round(
        total_time, 2
    )

    # ── Print Final Summary ───────────────
    print("\n" + "=" * 50)
    print("GreenBit — Pipeline Complete")
    print("=" * 50)
    print(
        f"  Total time     : "
        f"{total_time:.1f} seconds"
    )
    print(
        f"  Total files    : "
        f"{stats.get('total_files', 0)}"
    )
    print(
        f"  Exact dupes    : "
        f"{stats.get('exact_duplicate_count', 0)}"
    )
    print(
        f"  Clusters found : "
        f"{stats.get('clusters_found', 0)}"
    )
    print(
        f"  Reclaimable    : "
        f"{stats.get('reclaimable_gb', 0):.3f} GB"
    )
    print(
        f"  CO₂ emitted    : "
        f"{stats.get('co2_emissions', 0):.6f} kg"
    )
    print(
        f"  Net Env. Gain  : "
        f"{stats.get('net_environmental_gain_kwh', 0):.6f} kWh"
    )
    print(
        f"  Energy ROI     : "
        f"{stats.get('energy_roi', 0):.1f}:1"
    )
    print("=" * 50)

    return result_df, stats


# ── Quick Test ───────────────────────────────
# Run this file directly to test pipeline
# python run_pipeline.py

if __name__ == "__main__":
    import sys

    test_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "."
    )

    print("=" * 50)
    print("GreenBit — Pipeline Test")
    print("=" * 50)

    try:
        result_df, stats = stream_pipeline(
            test_path
        )

        print("\n── Final Statistics ────────────")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print(
            f"\n── DataFrame Shape: "
            f"{result_df.shape} ──"
        )

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()