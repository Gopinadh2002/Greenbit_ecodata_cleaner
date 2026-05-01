# ============================================
# reporting.py
# GreenBit Framework — Visualization and
# Reporting Module
# Responsibility:
#   - KPI metric card computation
#   - Six interactive Plotly charts
#   - Carbon equivalency calculations
#   - Before vs After comparison metrics
#   - CSV and JSON export generation
#   - Dashboard summary statistics
# ============================================

import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Color Palette ────────────────────────────

# GreenBit brand colors
COLOR_GREEN      = "#2ecc71"
COLOR_DARK_GREEN = "#27ae60"
COLOR_RED        = "#e74c3c"
COLOR_ORANGE     = "#f39c12"
COLOR_BLUE       = "#3498db"
COLOR_PURPLE     = "#9b59b6"
COLOR_TEAL       = "#1abc9c"
COLOR_DARK       = "#2c3e50"
COLOR_LIGHT      = "#ecf0f1"
COLOR_YELLOW     = "#f1c40f"

# Chart color sequences
CLUSTER_COLORS = [
    "#2ecc71", "#3498db", "#9b59b6",
    "#f39c12", "#e74c3c", "#1abc9c",
    "#e67e22", "#e91e63", "#00bcd4",
    "#8bc34a", "#ff5722", "#607d8b"
]

# Background and grid colors
CHART_BG    = "rgba(0,0,0,0)"
PAPER_BG    = "rgba(0,0,0,0)"
GRID_COLOR  = "rgba(255,255,255,0.1)"
FONT_COLOR  = "#ecf0f1"

# ── Constants ────────────────────────────────

BYTES_TO_GB = 1 / (1024 ** 3)
BYTES_TO_MB = 1 / (1024 ** 2)
BYTES_TO_KB = 1 / 1024

# Carbon intensity (kg CO₂ per kWh)
CARBON_INTENSITY = 0.475

# Real-world CO₂ equivalency factors
CO2_PER_TREE_PER_YEAR = 21.0  # kg CO₂/year
CO2_PER_KM_DRIVEN     = 0.21  # kg CO₂/km
CO2_PER_LED_HOUR      = 0.008 # kg CO₂/hour
CO2_PER_FLIGHT_HOUR   = 90.0  # kg CO₂/hour


# ── Helper Functions ─────────────────────────

def safe_get(stats, key, default=0):
    """
    Safely retrieve a value from the
    statistics dictionary with a default
    fallback value if key is missing
    or value is None.

    Args:
        stats (dict): Statistics dictionary
        key   (str) : Key to retrieve
        default     : Default fallback value

    Returns:
        Value from stats or default
    """
    value = stats.get(key, default)
    return value if value is not None else default


def format_size(size_bytes):
    """
    Format a byte size value into a
    human-readable string with appropriate
    unit (KB, MB, GB, TB).

    Args:
        size_bytes (int/float): Size in bytes

    Returns:
        str: Formatted size string
    """
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes * BYTES_TO_GB:.2f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes * BYTES_TO_MB:.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes * BYTES_TO_KB:.2f} KB"
    else:
        return f"{size_bytes:.0f} Bytes"


def format_time(seconds):
    """
    Format pipeline execution time
    into human-readable mm:ss string.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    else:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"


def get_chart_layout(title=""):
    """
    Return standardized Plotly layout
    configuration for all GreenBit charts.
    Ensures consistent dark theme styling
    across all visualizations.

    Args:
        title (str): Chart title text

    Returns:
        dict: Plotly layout configuration
    """
    return dict(
        title=dict(
            text=title,
            font=dict(
                color=FONT_COLOR,
                size=16,
                family="Arial"
            )
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font=dict(
            color=FONT_COLOR,
            family="Arial"
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True
        ),
        margin=dict(
            l=40, r=40, t=60, b=40
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor=GRID_COLOR,
            font=dict(color=FONT_COLOR)
        )
    )


# ── KPI Metrics Computation ──────────────────

def compute_kpi_metrics(df, stats):
    """
    Compute all Key Performance Indicator
    values required for the dashboard
    metric cards from the augmented
    DataFrame and statistics dictionary.

    Returns a structured dictionary
    containing all KPI values with
    appropriate formatting and units.

    Args:
        df    (pd.DataFrame): Augmented file
            metadata DataFrame
        stats (dict): Pipeline statistics

    Returns:
        dict: Complete KPI metrics dictionary
    """
    kpis = {}

    # ── Storage Metrics ───────────────────
    total_bytes = df["size"].sum() if not df.empty else 0
    kpis["total_storage_bytes"] = total_bytes
    kpis["total_storage_gb"]    = round(
        total_bytes * BYTES_TO_GB, 3
    )
    kpis["total_storage_fmt"]   = format_size(
        total_bytes
    )

    # ── File Count Metrics ────────────────
    kpis["total_files"] = len(df)

    # ── Exact Duplicate Metrics ───────────
    exact_dupes = df[
        df["is_duplicate"] == True
    ] if not df.empty else pd.DataFrame()

    kpis["exact_duplicate_count"] = len(
        exact_dupes
    )
    kpis["exact_duplicate_storage_fmt"] = (
        format_size(exact_dupes["size"].sum())
        if not exact_dupes.empty
        else "0 Bytes"
    )

    # ── Semantic Cluster Metrics ──────────
    if "cluster" in df.columns and not df.empty:
        waste_files = df[df["cluster"] >= 0]
        kpis["clusters_found"] = len(
            df[df["cluster"] >= 0]
            ["cluster"].unique()
        )
        kpis["semantic_waste_count"] = len(
            waste_files
        )
        kpis["semantic_waste_storage_fmt"] = (
            format_size(
                waste_files["size"].sum()
            )
        )
    else:
        kpis["clusters_found"]          = 0
        kpis["semantic_waste_count"]    = 0
        kpis["semantic_waste_storage_fmt"] = "0 Bytes"

    # ── Reclaimable Storage ───────────────
    reclaimable_bytes = safe_get(
        stats, "total_reclaimable_bytes", 0
    )
    kpis["reclaimable_storage_fmt"] = (
        format_size(reclaimable_bytes)
    )
    kpis["reclaimable_gb"] = round(
        reclaimable_bytes * BYTES_TO_GB, 3
    )

    # ── Average File Size ─────────────────
    avg_bytes = df["size"].mean() if not df.empty else 0
    kpis["avg_file_size_fmt"] = format_size(
        avg_bytes
    )

    # ── Carbon Metrics ────────────────────
    kpis["co2_emissions_kg"] = round(
        safe_get(stats, "co2_emissions", 0),
        6
    )
    kpis["net_gain_kwh"] = round(
        safe_get(
            stats,
            "net_environmental_gain_kwh",
            0
        ),
        6
    )
    kpis["energy_roi"] = round(
        safe_get(stats, "energy_roi", 0), 1
    )
    kpis["co2_offset_kg"] = round(
        safe_get(stats, "co2_offset_kg", 0),
        6
    )

    # ── Pipeline Time ─────────────────────
    pipeline_secs = safe_get(
        stats, "pipeline_time_seconds", 0
    )
    kpis["pipeline_time_fmt"] = format_time(
        pipeline_secs
    )

    return kpis


# ── Carbon Equivalency Calculations ──────────

def compute_carbon_equivalencies(co2_offset_kg):
    """
    Translate abstract CO₂ offset values
    into relatable real-world equivalents
    that non-technical users can understand
    and communicate in sustainability reports.

    Args:
        co2_offset_kg (float): CO₂ saved (kg)

    Returns:
        dict: Real-world equivalency metrics
    """
    equiv = {}

    # Trees equivalent (annual CO₂ absorption)
    equiv["trees_planted"] = round(
        co2_offset_kg / CO2_PER_TREE_PER_YEAR,
        4
    )

    # Car kilometers not driven
    equiv["km_not_driven"] = round(
        co2_offset_kg / CO2_PER_KM_DRIVEN,
        2
    )

    # LED bulb hours saved
    equiv["led_hours_saved"] = round(
        co2_offset_kg / CO2_PER_LED_HOUR,
        1
    )

    # Flight hours avoided
    equiv["flight_hours_avoided"] = round(
        co2_offset_kg / CO2_PER_FLIGHT_HOUR,
        4
    )

    return equiv


# ── Chart 1: File Size Distribution ──────────

def chart_file_size_distribution(df):
    """
    Generate a histogram showing the
    distribution of file sizes across
    the scanned directory.

    Reveals whether storage footprint is
    driven by many small files or fewer
    large files — critical insight for
    prioritizing cleanup strategy.

    Args:
        df (pd.DataFrame): File metadata

    Returns:
        plotly.graph_objects.Figure or None
    """
    if df is None or df.empty:
        return None

    try:
        plot_df = df.copy()
        plot_df["size_mb"] = (
            plot_df["size"] * BYTES_TO_MB
        )

        fig = px.histogram(
            plot_df,
            x="size_mb",
            nbins=50,
            title="📊 File Size Distribution",
            labels={
                "size_mb": "File Size (MB)",
                "count"  : "Number of Files"
            },
            color_discrete_sequence=[COLOR_BLUE]
        )

        fig.update_layout(
            **get_chart_layout(
                "📊 File Size Distribution"
            )
        )
        fig.update_traces(
            marker_line_color=COLOR_DARK,
            marker_line_width=0.5,
            opacity=0.85
        )

        return fig

    except Exception as e:
        print(f"[GreenBit] Chart error: {e}")
        return None


# ── Chart 2: Files per Semantic Cluster ──────

def chart_semantic_clusters(df):
    """
    Generate a bar chart showing the
    number of files in each semantic
    waste cluster identified by DBSCAN.

    Reveals the density and distribution
    of near-duplicate groups — the core
    output of GreenBit's AI engine.

    Args:
        df (pd.DataFrame): Augmented DataFrame
            with cluster column

    Returns:
        plotly.graph_objects.Figure or None
    """
    if df is None or df.empty:
        return None

    if "cluster" not in df.columns:
        return None

    try:
        # Include all clusters including noise
        cluster_counts = (
            df["cluster"]
            .value_counts()
            .reset_index()
        )
        cluster_counts.columns = [
            "Cluster ID", "File Count"
        ]

        # Sort by cluster ID
        cluster_counts = (
            cluster_counts
            .sort_values("Cluster ID")
        )

        # Label noise cluster clearly
        cluster_counts["Label"] = (
            cluster_counts["Cluster ID"]
            .apply(
                lambda x:
                "Unique (-1)" if x == -1
                else f"Exact Dupe (-2)" if x == -2
                else f"Cluster {x}"
            )
        )

        # Color coding
        cluster_counts["Color"] = (
            cluster_counts["Cluster ID"]
            .apply(
                lambda x:
                COLOR_TEAL if x == -1
                else COLOR_RED if x == -2
                else COLOR_PURPLE
            )
        )

        fig = px.bar(
            cluster_counts,
            x="Label",
            y="File Count",
            title="🔍 Files per Semantic Cluster",
            color="Label",
            color_discrete_sequence=(
                CLUSTER_COLORS
            )
        )

        fig.update_layout(
            **get_chart_layout(
                "🔍 Files per Semantic Cluster"
            ),
            showlegend=False
        )
        fig.update_traces(
            marker_line_color=COLOR_DARK,
            marker_line_width=0.5
        )

        return fig

    except Exception as e:
        print(f"[GreenBit] Chart error: {e}")
        return None


# ── Chart 3: Storage by File Type ────────────

def chart_storage_by_filetype(df):
    """
    Generate a pie chart showing the
    distribution of storage consumption
    across different file extensions.

    Identifies which file formats
    contribute most to the dark data
    footprint — enables targeted cleanup.

    Args:
        df (pd.DataFrame): File metadata

    Returns:
        plotly.graph_objects.Figure or None
    """
    if df is None or df.empty:
        return None

    try:
        plot_df = df.copy()

        # Get extension from path if not present
        if "extension" not in plot_df.columns:
            plot_df["extension"] = (
                plot_df["path"]
                .apply(
                    lambda x:
                    os.path.splitext(x)[1]
                    or "unknown"
                )
            )

        # Aggregate storage by extension
        storage = (
            plot_df
            .groupby("extension")["size"]
            .sum()
            .reset_index()
        )
        storage["size_gb"] = (
            storage["size"] * BYTES_TO_GB
        )
        storage = storage.sort_values(
            "size_gb", ascending=False
        )

        fig = px.pie(
            storage,
            values="size_gb",
            names="extension",
            title="💾 Storage Usage by File Type",
            color_discrete_sequence=(
                CLUSTER_COLORS
            ),
            hole=0.4  # Donut chart style
        )

        fig.update_layout(
            **get_chart_layout(
                "💾 Storage Usage by File Type"
            )
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(
                line=dict(
                    color=COLOR_DARK,
                    width=2
                )
            )
        )

        return fig

    except Exception as e:
        print(f"[GreenBit] Chart error: {e}")
        return None


# ── Chart 4: Files by Size Category ──────────

def chart_files_by_size_category(df):
    """
    Generate a donut pie chart categorizing
    all files into four size bands:
        Micro  : < 1 KB
        Small  : 1 KB — 1 MB
        Medium : 1 MB — 100 MB
        Large  : > 100 MB

    Provides structural view of directory
    composition by file size profile.

    Args:
        df (pd.DataFrame): File metadata

    Returns:
        plotly.graph_objects.Figure or None
    """
    if df is None or df.empty:
        return None

    try:
        plot_df = df.copy()

        # Categorize files by size
        def size_category(size_bytes):
            if size_bytes < 1024:
                return "Micro (< 1KB)"
            elif size_bytes < 1024 ** 2:
                return "Small (1KB–1MB)"
            elif size_bytes < 100 * 1024 ** 2:
                return "Medium (1MB–100MB)"
            else:
                return "Large (> 100MB)"

        plot_df["size_category"] = (
            plot_df["size"].apply(size_category)
        )

        category_counts = (
            plot_df["size_category"]
            .value_counts()
            .reset_index()
        )
        category_counts.columns = [
            "Category", "Count"
        ]

        fig = px.pie(
            category_counts,
            values="Count",
            names="Category",
            title="📁 Files by Size Category",
            color_discrete_sequence=[
                COLOR_TEAL,
                COLOR_BLUE,
                COLOR_ORANGE,
                COLOR_RED
            ],
            hole=0.4
        )

        fig.update_layout(
            **get_chart_layout(
                "📁 Files by Size Category"
            )
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(
                line=dict(
                    color=COLOR_DARK,
                    width=2
                )
            )
        )

        return fig

    except Exception as e:
        print(f"[GreenBit] Chart error: {e}")
        return None


# ── Chart 5: Top 10 Largest Files ────────────

def chart_top_largest_files(df):
    """
    Generate a horizontal bar chart
    ranking the top 10 largest files
    in the scanned directory by size.

    Enables targeted manual review of
    highest-impact files before deletion
    to prevent accidental data loss.

    Args:
        df (pd.DataFrame): File metadata

    Returns:
        plotly.graph_objects.Figure or None
    """
    if df is None or df.empty:
        return None

    try:
        plot_df = df.copy()
        plot_df["size_mb"] = (
            plot_df["size"] * BYTES_TO_MB
        )

        # Get filename for display
        if "filename" not in plot_df.columns:
            plot_df["filename"] = (
                plot_df["path"]
                .apply(os.path.basename)
            )

        # Get top 10 largest files
        top10 = (
            plot_df
            .nlargest(10, "size_mb")
            .copy()
        )

        # Truncate long filenames
        top10["display_name"] = (
            top10["filename"]
            .apply(
                lambda x:
                x[:40] + "..."
                if len(x) > 40
                else x
            )
        )

        fig = px.bar(
            top10,
            x="size_mb",
            y="display_name",
            orientation="h",
            title="🏆 Top 10 Largest Files",
            labels={
                "size_mb"     : "Size (MB)",
                "display_name": "File Name"
            },
            color="size_mb",
            color_continuous_scale=[
                COLOR_GREEN,
                COLOR_ORANGE,
                COLOR_RED
            ]
        )

        fig.update_layout(
            **get_chart_layout(
                "🏆 Top 10 Largest Files"
            ),
            yaxis=dict(
                autorange="reversed",
                gridcolor=GRID_COLOR
            ),
            coloraxis_showscale=False
        )

        return fig

    except Exception as e:
        print(f"[GreenBit] Chart error: {e}")
        return None


# ── Chart 6: Energy ROI Waterfall ────────────

def chart_energy_waterfall(stats):
    """
    Generate a waterfall chart showing
    the energy accounting breakdown:
        Energy Consumed (audit cost)
        Energy Saved   (storage savings)
        Net Environmental Gain

    Provides clear visual proof of
    GreenBit's positive ecological outcome.

    Args:
        stats (dict): Pipeline statistics

    Returns:
        plotly.graph_objects.Figure or None
    """
    try:
        audit_energy = safe_get(
            stats, "audit_energy_kwh", 0
        )
        energy_saved = safe_get(
            stats, "annual_energy_saved_kwh", 0
        )
        net_gain = safe_get(
            stats,
            "net_environmental_gain_kwh",
            0
        )

        fig = go.Figure(go.Waterfall(
            name="Energy Flow",
            orientation="v",
            measure=[
                "relative",
                "relative",
                "total"
            ],
            x=[
                "Energy Consumed\n(Audit Cost)",
                "Energy Saved\n(Storage Reclaim)",
                "Net Environmental\nGain"
            ],
            y=[
                -audit_energy,
                energy_saved,
                net_gain
            ],
            text=[
                f"-{audit_energy:.6f} kWh",
                f"+{energy_saved:.6f} kWh",
                f"{net_gain:.6f} kWh"
            ],
            textposition="outside",
            connector=dict(
                line=dict(
                    color=GRID_COLOR,
                    width=1
                )
            ),
            decreasing=dict(
                marker=dict(color=COLOR_RED)
            ),
            increasing=dict(
                marker=dict(color=COLOR_GREEN)
            ),
            totals=dict(
                marker=dict(color=COLOR_TEAL)
            )
        ))

        fig.update_layout(
            **get_chart_layout(
                "⚡ Energy ROI Waterfall"
            ),
            showlegend=False
        )

        return fig

    except Exception as e:
        print(f"[GreenBit] Chart error: {e}")
        return None


# ── Chart 7: Environmental Gauge ─────────────

def chart_environmental_gauge(stats):
    """
    Generate a speedometer gauge showing
    the Net Environmental Gain as a
    visual health score indicator.

    Color zones:
        Red    : Negative gain (harmful)
        Yellow : Low positive gain
        Green  : High positive gain

    Args:
        stats (dict): Pipeline statistics

    Returns:
        plotly.graph_objects.Figure or None
    """
    try:
        energy_roi = safe_get(
            stats, "energy_roi", 0
        )

        # Cap display at 1000 for readability
        display_roi = min(energy_roi, 1000)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=display_roi,
            title=dict(
                text="🌿 Energy ROI Score",
                font=dict(
                    color=FONT_COLOR,
                    size=18
                )
            ),
            number=dict(
                suffix=":1",
                font=dict(
                    color=FONT_COLOR,
                    size=28
                )
            ),
            delta=dict(
                reference=10,
                increasing=dict(
                    color=COLOR_GREEN
                ),
                decreasing=dict(
                    color=COLOR_RED
                )
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 1000],
                    tickwidth=1,
                    tickcolor=FONT_COLOR,
                    tickfont=dict(
                        color=FONT_COLOR
                    )
                ),
                bar=dict(
                    color=COLOR_GREEN,
                    thickness=0.3
                ),
                bgcolor="rgba(0,0,0,0.2)",
                borderwidth=2,
                bordercolor=GRID_COLOR,
                steps=[
                    dict(
                        range=[0, 1],
                        color=COLOR_RED
                    ),
                    dict(
                        range=[1, 10],
                        color=COLOR_ORANGE
                    ),
                    dict(
                        range=[10, 100],
                        color=COLOR_YELLOW
                    ),
                    dict(
                        range=[100, 1000],
                        color=COLOR_GREEN
                    )
                ],
                threshold=dict(
                    line=dict(
                        color=COLOR_TEAL,
                        width=4
                    ),
                    thickness=0.75,
                    value=energy_roi
                )
            )
        ))

        fig.update_layout(
            paper_bgcolor=PAPER_BG,
            plot_bgcolor=CHART_BG,
            font=dict(color=FONT_COLOR),
            margin=dict(
                l=30, r=30, t=60, b=30
            )
        )

        return fig

    except Exception as e:
        print(f"[GreenBit] Chart error: {e}")
        return None


# ── Before vs After Comparison ───────────────

def compute_before_after(df, stats):
    """
    Calculate Before and After metrics
    showing the impact of removing all
    identified redundant files from
    the scanned directory.

    Args:
        df    (pd.DataFrame): File metadata
        stats (dict): Pipeline statistics

    Returns:
        dict: Before and After metric pairs
    """
    total_bytes = df["size"].sum() if not df.empty else 0
    reclaimable = safe_get(
        stats, "total_reclaimable_bytes", 0
    )
    after_bytes = max(
        total_bytes - reclaimable, 0
    )

    total_files = len(df)
    waste_count = (
        safe_get(stats, "exact_duplicate_count", 0)
        + safe_get(stats, "semantic_waste_count", 0)
    )
    after_files = max(
        total_files - waste_count, 0
    )

    # CO₂ calculations
    co2_before = (
        total_bytes * BYTES_TO_GB
        * 0.000002 * CARBON_INTENSITY * 1000
    )
    co2_after = (
        after_bytes * BYTES_TO_GB
        * 0.000002 * CARBON_INTENSITY * 1000
    )

    return {
        "before_storage_fmt" : format_size(total_bytes),
        "after_storage_fmt"  : format_size(after_bytes),
        "before_files"       : total_files,
        "after_files"        : after_files,
        "before_co2_g"       : round(co2_before, 4),
        "after_co2_g"        : round(co2_after, 4),
        "savings_storage_fmt": format_size(reclaimable),
        "savings_files"      : waste_count,
        "savings_pct"        : round(
            (reclaimable / total_bytes * 100)
            if total_bytes > 0 else 0,
            1
        )
    }


# ── Export Functions ─────────────────────────

def export_csv(df, output_path="outputs/audit_report.csv"):
    """
    Export the complete audit DataFrame
    to a UTF-8 encoded CSV file suitable
    for import into ESG reporting platforms
    and spreadsheet applications.

    Columns exported:
        path, filename, extension,
        size, hash, is_duplicate, cluster

    Args:
        df          (pd.DataFrame): Audit data
        output_path (str): Output file path

    Returns:
        str: Path to exported CSV file
             or None on failure
    """
    try:
        # Ensure output directory exists
        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        # Select and order export columns
        export_cols = [
            col for col in [
                "path", "filename",
                "extension", "size",
                "hash", "is_duplicate",
                "cluster"
            ] if col in df.columns
        ]

        export_df = df[export_cols].copy()

        # Add human-readable size column
        export_df["size_mb"] = (
            export_df["size"] * BYTES_TO_MB
        ).round(4)

        # Add cluster label column
        if "cluster" in export_df.columns:
            export_df["cluster_label"] = (
                export_df["cluster"].apply(
                    lambda x:
                    "Unique" if x == -1
                    else "Exact Duplicate"
                    if x == -2
                    else f"Waste Cluster {x}"
                )
            )

        export_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8"
        )

        print(
            f"[GreenBit] CSV exported: "
            f"{output_path}"
        )
        return output_path

    except Exception as e:
        print(
            f"[GreenBit] CSV export "
            f"error: {e}"
        )
        return None


def export_json(
    df,
    stats,
    output_path="outputs/audit_report.json"
):
    """
    Export audit results to a structured
    JSON file containing both the complete
    file records and the pipeline statistics
    summary for programmatic processing by
    downstream data management systems.

    Args:
        df          (pd.DataFrame): Audit data
        stats       (dict): Pipeline statistics
        output_path (str): Output file path

    Returns:
        str: Path to exported JSON file
             or None on failure
    """
    try:
        # Ensure output directory exists
        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        # Build export dictionary
        export_data = {
            "audit_summary" : stats,
            "file_records"  : (
                df.to_dict(orient="records")
                if not df.empty
                else []
            )
        }

        with open(
            output_path, "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                export_data, f,
                indent=2,
                default=str  # Handle non-serializable types
            )

        print(
            f"[GreenBit] JSON exported: "
            f"{output_path}"
        )
        return output_path

    except Exception as e:
        print(
            f"[GreenBit] JSON export "
            f"error: {e}"
        )
        return None


def get_csv_bytes(df):
    """
    Generate CSV content as bytes for
    Streamlit download button without
    saving to disk.

    Args:
        df (pd.DataFrame): Audit DataFrame

    Returns:
        bytes: UTF-8 encoded CSV content
    """
    try:
        export_cols = [
            col for col in [
                "path", "filename",
                "extension", "size",
                "hash", "is_duplicate",
                "cluster"
            ] if col in df.columns
        ]

        export_df = df[export_cols].copy()
        export_df["size_mb"] = (
            export_df["size"] * BYTES_TO_MB
        ).round(4)

        if "cluster" in export_df.columns:
            export_df["cluster_label"] = (
                export_df["cluster"].apply(
                    lambda x:
                    "Unique" if x == -1
                    else "Exact Duplicate"
                    if x == -2
                    else f"Waste Cluster {x}"
                )
            )

        return export_df.to_csv(
            index=False
        ).encode("utf-8")

    except Exception:
        return b""


def get_json_bytes(df, stats):
    """
    Generate JSON content as bytes for
    Streamlit download button without
    saving to disk.

    Args:
        df    (pd.DataFrame): Audit DataFrame
        stats (dict): Pipeline statistics

    Returns:
        bytes: UTF-8 encoded JSON content
    """
    try:
        export_data = {
            "audit_summary": stats,
            "file_records" : (
                df.to_dict(orient="records")
                if not df.empty
                else []
            )
        }
        return json.dumps(
            export_data,
            indent=2,
            default=str
        ).encode("utf-8")

    except Exception:
        return b"{}"


# ── All Charts Generator ─────────────────────

def generate_all_charts(df, stats):
    """
    Generate all GreenBit visualization
    charts and return them as a dictionary
    for rendering on the Streamlit dashboard.

    Args:
        df    (pd.DataFrame): Augmented file
            metadata DataFrame
        stats (dict): Pipeline statistics

    Returns:
        dict: Dictionary of Plotly figures
            keyed by chart name
    """
    charts = {}

    charts["file_size_dist"] = (
        chart_file_size_distribution(df)
    )
    charts["semantic_clusters"] = (
        chart_semantic_clusters(df)
    )
    charts["storage_by_type"] = (
        chart_storage_by_filetype(df)
    )
    charts["size_categories"] = (
        chart_files_by_size_category(df)
    )
    charts["top_largest"] = (
        chart_top_largest_files(df)
    )
    charts["energy_waterfall"] = (
        chart_energy_waterfall(stats)
    )
    charts["env_gauge"] = (
        chart_environmental_gauge(stats)
    )

    return charts