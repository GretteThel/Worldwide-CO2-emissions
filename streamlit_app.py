from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events2 import plotly_events

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Worldwide CO₂ Emissions Dashboard", layout="wide")

# ----------------------------
# Theme-aware UI colors
# ----------------------------
THEME_BASE = st.get_option("theme.base") or "light"

TEXT = "#F8FAFC" if THEME_BASE == "dark" else "#0F172A"
BORDER = "#334155" if THEME_BASE == "dark" else "#E2E8F0"
MUTED = "#CBD5E1" if THEME_BASE == "dark" else "#475569"
SOFT = "#94A3B8" if THEME_BASE == "dark" else "#64748B"

# ----------------------------
# Paths and constants
# ----------------------------
HERE = Path(__file__).resolve().parent
COUNTRY_YEAR_FILE = HERE / "co2_country_year_merged.csv"
SECTOR_FILE = HERE / "co2_sector_country_long.csv"
CENTROIDS_FILE = HERE / "country_centroids.csv"

ACCENT = "#2563EB"
HIGHLIGHT = "#F59E0B"

GRAPH_CONFIG_MAP = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "scrollZoom": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "toImage",
        "lasso2d",
        "select2d",
        "autoScale2d",
        "toggleSpikelines",
    ],
}

GRAPH_CONFIG_SELECT = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "scrollZoom": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "toImage",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "toggleSpikelines",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "pan2d",
    ],
}

GRAPH_CONFIG_VIEW_ONLY = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "scrollZoom": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["toImage", "toggleSpikelines"],
}

PREFERRED_SECTOR_ORDER = [
    "Power Industry",
    "Transport",
    "Other industrial combustion",
    "Buildings",
    "Other sectors",
]


# ----------------------------
# Helpers
# ----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return out


def chart_title(text: str, size: int = 16) -> dict:
    return {
        "text": f"<b>{text}</b>",
        "x": 0.02,
        "xanchor": "left",
        "y": 0.98,
        "yanchor": "top",
        "font": {"size": size},
    }


def empty_figure(title: str, x_title: str = "", y_title: str = "", height: int = 320):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title=chart_title(title, 18),
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        margin=dict(l=10, r=16, t=78, b=20),
        annotations=[
            {
                "text": "No data available for the current filters",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14, "color": "#64748B"},
            }
        ],
    )
    return fig


def get_bubble_sizes(series: pd.Series, min_size: float = 8, max_size: float = 32):
    if series.empty:
        return []
    values = series.fillna(0).astype(float)
    if values.max() == values.min():
        return np.full(len(values), (min_size + max_size) / 2)
    scaled = (values - values.min()) / (values.max() - values.min())
    return min_size + scaled * (max_size - min_size)


def choose_sector_palette(sectors):
    base = {
        "Power Industry": "#0072B2",
        "Transport": "#E69F00",
        "Other industrial combustion": "#009E73",
        "Buildings": "#CC79A7",
        "Other sectors": "#999999",
    }
    palette = {}
    extras = ["#56B4E9", "#D55E00", "#F0E442"]
    extra_idx = 0
    for sector in sectors:
        if sector in base:
            palette[sector] = base[sector]
        else:
            palette[sector] = extras[extra_idx % len(extras)]
            extra_idx += 1
    return palette


def load_centroids(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["country_code", "lat", "lon"])
    cent = normalize_columns(pd.read_csv(path))
    required = {"country_code", "lat", "lon"}
    if not required.issubset(set(cent.columns)):
        return pd.DataFrame(columns=["country_code", "lat", "lon"])
    cent = cent[list(required)].copy()
    cent["country_code"] = cent["country_code"].astype(str).str.strip().str.upper()
    cent["lat"] = pd.to_numeric(cent["lat"], errors="coerce")
    cent["lon"] = pd.to_numeric(cent["lon"], errors="coerce")
    cent = cent.dropna(subset=["country_code", "lat", "lon"]).drop_duplicates(subset=["country_code"])
    return cent


def build_comparison_view(
    year_df: pd.DataFrame,
    selected_countries: list[str],
    selected_country_codes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_dff = year_df.copy()
    if selected_countries:
        filtered_dff = filtered_dff[filtered_dff["country"].isin(selected_countries)].copy()

    dff = filtered_dff.copy()
    if selected_country_codes:
        extra_rows = year_df[year_df["country_code"].isin(selected_country_codes)].copy()
        dff = pd.concat([dff, extra_rows], ignore_index=True)
        dff = dff.drop_duplicates(subset=["country_code"]).copy()

    return filtered_dff, dff


def build_sector_comparison_view(
    sector_year_df: pd.DataFrame,
    selected_countries: list[str],
    selected_country_codes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_sector_df = sector_year_df.copy()
    if selected_countries:
        filtered_sector_df = filtered_sector_df[filtered_sector_df["country"].isin(selected_countries)].copy()

    sector_dff = filtered_sector_df.copy()
    if selected_country_codes:
        extra_rows = sector_year_df[sector_year_df["country_code"].isin(selected_country_codes)].copy()
        sector_dff = pd.concat([sector_dff, extra_rows], ignore_index=True)
        dedupe_cols = [col for col in ["country_code", "country", "sector", "year"] if col in sector_dff.columns]
        if dedupe_cols:
            sector_dff = sector_dff.drop_duplicates(subset=dedupe_cols).copy()
        else:
            sector_dff = sector_dff.drop_duplicates().copy()

    return filtered_sector_df, sector_dff


def bar_title_text(
    filtered_count: int,
    requested_top_n: int,
    actual_shown_count: int,
    selected_year: int,
    selected_outside_topn: bool,
    selected_count: int,
) -> str:
    if filtered_count <= 1 and selected_count == 0:
        return f"Emissions in current filter ({selected_year})"
    if selected_outside_topn:
        suffix = "countries" if selected_count > 1 else "country"
        return f"Largest fossil CO₂ emitters + selected {suffix} ({selected_year})"
    return f"Largest fossil CO₂ emitters ({selected_year})"


def sector_title_text(
    selected_year: int,
    selected_names: list[str],
    has_filter_focus: bool,
    shown_count: int,
) -> str:
    if selected_names and not has_filter_focus:
        if len(selected_names) == 1:
            return f"Sector contributions for {selected_names[0]} ({selected_year})"
        return f"Sector contributions for selected countries ({selected_year})"
    if has_filter_focus and selected_names:
        return f"Sector contributions for current focus ({selected_year})"
    if has_filter_focus:
        return f"Sector contributions for filtered countries ({selected_year})"
    return f"Sector contributions to fossil CO₂ emissions ({selected_year})"


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def get_selection_points(event) -> list[dict]:
    if not event:
        return []

    selection = None
    if isinstance(event, dict):
        selection = event.get("selection")
    else:
        selection = getattr(event, "selection", None)

    if selection is None:
        return []

    if isinstance(selection, dict):
        return selection.get("points", []) or []

    points = getattr(selection, "points", None)
    if points is None and hasattr(selection, "get"):
        points = selection.get("points", [])
    return points or []


def extract_codes_from_selection_points(points: list[dict]) -> list[str]:
    codes = []
    for point in points:
        custom = point.get("customdata")
        if isinstance(custom, (list, tuple, np.ndarray)):
            code = custom[0] if len(custom) > 0 else None
        else:
            code = custom
        if code is not None:
            codes.append(str(code).strip().upper())
    return unique_preserve_order(codes)


def toggle_codes(codes: list[str]) -> bool:
    codes = unique_preserve_order([str(code).strip().upper() for code in codes if code])
    if not codes:
        return False

    selected_codes = list(st.session_state.get("selected_country_codes", []))
    selected_set = set(selected_codes)
    codes_set = set(codes)

    if codes_set.issubset(selected_set):
        selected_codes = [code for code in selected_codes if code not in codes_set]
    else:
        for code in codes:
            if code not in selected_set:
                selected_codes.append(code)
                selected_set.add(code)

    st.session_state["selected_country_codes"] = selected_codes
    return True


def process_native_selection(chart_name: str, event) -> bool:
    points = get_selection_points(event)
    codes = extract_codes_from_selection_points(points)
    sig_key = f"last_{chart_name}_selection_sig"
    sig = tuple(codes)

    if not codes:
        st.session_state[sig_key] = None
        return False

    if st.session_state.get(sig_key) == sig:
        return False

    st.session_state[sig_key] = sig
    return toggle_codes(codes)


def extract_country_code_from_map_event(points: list, lookups: dict) -> Optional[str]:
    if not points:
        return None

    point = points[0]
    curve = point.get("curveNumber", 0)
    idx = point.get("pointIndex", point.get("pointNumber"))
    trace_meta = lookups.get("map_trace_meta", [])

    if idx is None or curve is None or curve < 0 or curve >= len(trace_meta):
        return None

    _, trace_df = trace_meta[curve]
    if 0 <= idx < len(trace_df):
        return trace_df.iloc[idx]["country_code"]
    return None


def process_map_click(points: list, lookups: dict) -> bool:
    if not points:
        return False

    point = points[0]
    click_sig = (
        point.get("curveNumber"),
        point.get("pointIndex", point.get("pointNumber")),
        point.get("x"),
        point.get("y"),
    )

    if st.session_state.get("last_map_click_sig") == click_sig:
        return False

    st.session_state["last_map_click_sig"] = click_sig
    code = extract_country_code_from_map_event(points, lookups)
    return toggle_codes([code]) if code else False


@st.cache_data
def load_data():
    if not COUNTRY_YEAR_FILE.exists():
        raise FileNotFoundError("Missing co2_country_year_merged.csv in the same folder as streamlit_app.py")
    if not SECTOR_FILE.exists():
        raise FileNotFoundError("Missing co2_sector_country_long.csv in the same folder as streamlit_app.py")

    country_year = normalize_columns(pd.read_csv(COUNTRY_YEAR_FILE))
    sector_long = normalize_columns(pd.read_csv(SECTOR_FILE))
    centroids = load_centroids(CENTROIDS_FILE)

    country_year = country_year.dropna(subset=["country_code", "country", "year"]).copy()
    sector_long = sector_long.dropna(subset=["country_code", "country", "year"]).copy()

    country_year["country_code"] = country_year["country_code"].astype(str).str.strip().str.upper()
    sector_long["country_code"] = sector_long["country_code"].astype(str).str.strip().str.upper()

    country_year["country"] = country_year["country"].astype(str).str.strip()
    sector_long["country"] = sector_long["country"].astype(str).str.strip()

    country_year["year"] = pd.to_numeric(country_year["year"], errors="coerce").astype("Int64")
    sector_long["year"] = pd.to_numeric(sector_long["year"], errors="coerce").astype("Int64")

    for col in ["total_mt_co2", "co2_per_gdp_t_per_kusd", "co2_per_capita_t", "log_total_mt_co2"]:
        if col in country_year.columns:
            country_year[col] = pd.to_numeric(country_year[col], errors="coerce")

    if "sector_mt_co2" in sector_long.columns:
        sector_long["sector_mt_co2"] = pd.to_numeric(sector_long["sector_mt_co2"], errors="coerce")

    if "entity_type" in country_year.columns:
        country_year = country_year[country_year["entity_type"] == "country"].copy()

    if "entity_type" in sector_long.columns:
        sector_long = sector_long[sector_long["entity_type"] == "country"].copy()

    country_year = country_year[country_year["year"].notna()].copy()
    sector_long = sector_long[sector_long["year"].notna()].copy()

    country_year["year"] = country_year["year"].astype(int)
    sector_long["year"] = sector_long["year"].astype(int)

    return country_year, sector_long, centroids


# ----------------------------
# Load data
# ----------------------------
country_year, sector_long, centroids = load_data()
YEARS = sorted(country_year["year"].dropna().unique().tolist())
COUNTRIES = sorted(country_year["country"].dropna().unique().tolist())


# ----------------------------
# Session state
# ----------------------------
for key, default in {
    "year": max(YEARS),
    "countries": [],
    "top_n": 10,
    "selected_country_codes": [],
    "last_map_click_sig": None,
    "last_bar_selection_sig": None,
    "last_scatter_selection_sig": None,
    "ui_revision": 0,
    "skip_event_processing_once": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset_all():
    st.session_state["year"] = max(YEARS)
    st.session_state["countries"] = []
    st.session_state["top_n"] = 10
    st.session_state["selected_country_codes"] = []
    st.session_state["last_map_click_sig"] = None
    st.session_state["last_bar_selection_sig"] = None
    st.session_state["last_scatter_SELECTION_SIG"] = None
    st.session_state["last_scatter_selection_sig"] = None
    st.session_state["ui_revision"] += 1
    st.session_state["skip_event_processing_once"] = True


# ----------------------------
# Page styles
# ----------------------------
st.markdown(
    f"""
    <style>
    .block-container {{
        padding-top: 2.15rem;
        padding-bottom: 1.6rem;
        max-width: 1500px;
    }}
    .status-box {{
        border-top: 1px solid {BORDER};
        margin-top: 1rem;
        padding-top: 1rem;
        color: {MUTED};
        font-size: 0.92rem;
        white-space: pre-line;
    }}
    .small-note {{
        color: {SOFT};
        font-size: 0.78rem;
        margin-top: 0.75rem;
    }}
    div[data-testid="stExpander"] details summary p {{
        font-size: 0.88rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Worldwide CO₂ Emissions Dashboard")
st.write("Explore which countries emit the most, how carbon-intensive they are, and which sectors explain those differences.")
st.caption(
    "Tip: click the map to toggle countries, and use click, box select, or lasso select on the bar and scatter charts to update the whole dashboard. Hover reveals country names only when needed, keeping the view clean."
)


def emitters_marks_html(selected_value: int) -> str:
    marks = [5, 10, 15, 20]
    chunks = []
    for value in marks:
        color = TEXT if value == selected_value else SOFT
        weight = 700 if value == selected_value else 500
        chunks.append(
            f'<span style="color:{color};font-weight:{weight};min-width:26px;text-align:center;display:inline-block;">{value}</span>'
        )
    return (
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'margin:-0.45rem 0 0.35rem 0;padding:0 2px;font-size:0.82rem;">'
        + "".join(chunks)
        + "</div>"
    )


# ----------------------------
# Layout
# ----------------------------
revision = st.session_state["ui_revision"]
sidebar_col, main_col = st.columns([0.24, 0.76], vertical_alignment="top")

with sidebar_col:
    st.markdown("## Filters")
    st.selectbox("Year", YEARS, key="year")
    st.multiselect("Country", COUNTRIES, key="countries", placeholder="All countries")
    st.slider("Top emitters to show", min_value=5, max_value=20, step=1, key="top_n")
    st.markdown(emitters_marks_html(st.session_state["top_n"]), unsafe_allow_html=True)
    st.button("Reset filters and focus", use_container_width=True, on_click=reset_all)

selected_year = st.session_state["year"]
year_df = country_year[country_year["year"] == selected_year].copy()
year_df = year_df.dropna(subset=["country_code", "country"]).copy()

available_year_codes = set(year_df["country_code"].tolist())
selected_country_codes = [
    code for code in st.session_state["selected_country_codes"] if code in available_year_codes
]
if selected_country_codes != st.session_state["selected_country_codes"]:
    st.session_state["selected_country_codes"] = selected_country_codes

active_country_code = selected_country_codes[-1] if selected_country_codes else None
filtered_dff, dff = build_comparison_view(
    year_df=year_df,
    selected_countries=st.session_state["countries"],
    selected_country_codes=selected_country_codes,
)

country_filter_codes = unique_preserve_order(
    year_df.loc[year_df["country"].isin(st.session_state["countries"]), "country_code"]
    .dropna()
    .astype(str)
    .str.upper()
    .tolist()
)
country_filter_set = set(country_filter_codes)
country_filter_active = len(country_filter_codes) > 0

selected_country_names = (
    year_df.loc[year_df["country_code"].isin(selected_country_codes), ["country_code", "country"]]
    .drop_duplicates()
    .set_index("country_code")
    .reindex(selected_country_codes)["country"]
    .dropna()
    .tolist()
)

bar_rank_df = (
    dff.dropna(subset=["total_mt_co2"])
    .copy()
    .sort_values("total_mt_co2", ascending=False)
    .reset_index(drop=True)
)
bar_rank_df["rank_total_emissions"] = np.arange(1, len(bar_rank_df) + 1)
rank_lookup = dict(zip(bar_rank_df["country_code"], bar_rank_df["rank_total_emissions"]))
selected_rank = rank_lookup.get(active_country_code) if active_country_code else None
selected_outside_topn = any(rank_lookup.get(code, 10**9) > st.session_state["top_n"] for code in selected_country_codes)
selected_outside_filter = country_filter_active and any(code not in country_filter_set for code in selected_country_codes)

scatter_df_for_status = dff.dropna(subset=["co2_per_gdp_t_per_kusd", "co2_per_capita_t", "total_mt_co2"]).copy()
scatter_available = scatter_df_for_status["country"].nunique() if not scatter_df_for_status.empty else 0

selected_names_text = ", ".join(selected_country_names) if selected_country_names else "None"
latest_rank_line = (
    f"Latest selected country rank by total emissions: {selected_rank}"
    if selected_rank is not None
    else "Latest selected country rank by total emissions: None"
)
selection_note = (
    "Selected countries outside the country filter are included in the comparison view."
    if selected_outside_filter
    else "All selected countries are inside the current country filter."
    if selected_country_codes and country_filter_active
    else "A country filter is not active; selected countries define the current focus."
    if selected_country_codes
    else "No country selected from the visuals."
)
outside_top_note = (
    f"Selected countries outside the top {st.session_state['top_n']} emitters have been appended to the bar chart."
    if selected_outside_topn
    else f"All selected countries are within the top {st.session_state['top_n']} emitters."
    if selected_country_codes
    else ""
)

with sidebar_col:
    st.markdown(
        f"""
        <div class="status-box">
        Year: {selected_year}
        <br>Countries in comparison view: {dff['country'].nunique():,}
        <br>Scatter countries with complete intensity data: {scatter_available:,}
        <br>Selected countries: {selected_names_text}
        <br>{latest_rank_line}
        <br>{selection_note}
        <br>{outside_top_note}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="small-note">
        Color guide: the blue scale next to the map belongs only to the choropleth and represents CO₂ per capita (t CO₂/cap/yr). In the other charts, blue shows the comparison view and orange highlights the selected countries. Country names appear on hover only, to keep the visuals consistent and uncluttered.
        <br><br>
        Interactions are linked across the dashboard: the country filter defines the base comparison set, while clicks, box select, and lasso select add a focused layer on top. The sector chart follows that same focus so users can compare selected countries with the countries already in the filter.
        </div>
        """,
        unsafe_allow_html=True,
    )

with main_col:
    # ----------------------------
    # Map
    # ----------------------------
    map_df = year_df.dropna(subset=["co2_per_capita_t"]).copy()
    filter_codes = set(
        year_df.loc[year_df["country"].isin(st.session_state["countries"]), "country_code"]
        .dropna()
        .astype(str)
        .str.upper()
        .tolist()
    )
    selected_codes_set = set(selected_country_codes)
    filter_only_codes = filter_codes - selected_codes_set

    map_filter_outline_df = map_df[map_df["country_code"].isin(filter_only_codes)].copy()
    map_selected_df = map_df[map_df["country_code"].isin(selected_codes_set)].copy()

    map_marker_df = map_df[["country_code", "country"]].drop_duplicates().copy()
    map_marker_df = map_marker_df.merge(centroids, on="country_code", how="left").dropna(subset=["lat", "lon"])

    map_selected_marker_df = map_selected_df[["country_code", "country"]].drop_duplicates().copy()
    map_selected_marker_df = map_selected_marker_df.merge(centroids, on="country_code", how="left").dropna(subset=["lat", "lon"])

    map_filter_marker_df = map_filter_outline_df[["country_code", "country"]].drop_duplicates().copy()
    map_filter_marker_df = map_filter_marker_df.merge(centroids, on="country_code", how="left").dropna(subset=["lat", "lon"])

    map_trace_meta: list[tuple[str, pd.DataFrame]] = []

    if map_df.empty:
        map_fig = empty_figure("Fossil CO₂ per capita by country", height=470)
    else:
        map_fig = go.Figure()
        map_fig.add_trace(
            go.Choropleth(
                locations=map_df["country_code"],
                z=map_df["co2_per_capita_t"],
                locationmode="ISO-3",
                colorscale="Blues",
                colorbar=dict(
                    title="Map scale<br>CO₂ per capita<br>(t CO₂/cap/yr)",
                    len=0.82,
                    y=0.5,
                    thickness=18,
                ),
                marker_line_color="white",
                marker_line_width=0.5,
                customdata=np.stack(
                    [
                        map_df["country_code"],
                        map_df["country"],
                        map_df["total_mt_co2"].fillna(np.nan),
                        map_df["co2_per_gdp_t_per_kusd"].fillna(np.nan),
                        map_df["co2_per_capita_t"].fillna(np.nan),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "Total emissions: %{customdata[2]:,.1f} Mt CO₂/yr<br>"
                    "CO₂ per capita: %{customdata[4]:,.2f} t CO₂/cap/yr<br>"
                    "CO₂ per GDP: %{customdata[3]:,.2f} t CO₂/kUSD/yr"
                    "<extra></extra>"
                ),
            )
        )
        map_trace_meta.append(("base", map_df.reset_index(drop=True)))

        if not map_filter_outline_df.empty:
            map_fig.add_trace(
                go.Choropleth(
                    locations=map_filter_outline_df["country_code"],
                    z=np.ones(len(map_filter_outline_df)),
                    locationmode="ISO-3",
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                    showscale=False,
                    marker_line_color=ACCENT,
                    marker_line_width=1.8,
                    customdata=np.stack(
                        [
                            map_filter_outline_df["country_code"],
                            map_filter_outline_df["country"],
                            map_filter_outline_df["total_mt_co2"].fillna(np.nan),
                            map_filter_outline_df["co2_per_gdp_t_per_kusd"].fillna(np.nan),
                            map_filter_outline_df["co2_per_capita_t"].fillna(np.nan),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "Country filter selection<br>"
                        "Total emissions: %{customdata[2]:,.1f} Mt CO₂/yr<br>"
                        "CO₂ per capita: %{customdata[4]:,.2f} t CO₂/cap/yr<br>"
                        "CO₂ per GDP: %{customdata[3]:,.2f} t CO₂/kUSD/yr"
                        "<extra></extra>"
                    ),
                )
            )
            map_trace_meta.append(("filter_outline", map_filter_outline_df.reset_index(drop=True)))

        if not map_selected_df.empty:
            map_fig.add_trace(
                go.Choropleth(
                    locations=map_selected_df["country_code"],
                    z=np.ones(len(map_selected_df)),
                    locationmode="ISO-3",
                    colorscale=[[0, HIGHLIGHT], [1, HIGHLIGHT]],
                    showscale=False,
                    marker_line_color=HIGHLIGHT,
                    marker_line_width=2.3,
                    customdata=np.stack(
                        [
                            map_selected_df["country_code"],
                            map_selected_df["country"],
                            map_selected_df["total_mt_co2"].fillna(np.nan),
                            map_selected_df["co2_per_gdp_t_per_kusd"].fillna(np.nan),
                            map_selected_df["co2_per_capita_t"].fillna(np.nan),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "Selected country<br>"
                        "Total emissions: %{customdata[2]:,.1f} Mt CO₂/yr<br>"
                        "CO₂ per capita: %{customdata[4]:,.2f} t CO₂/cap/yr<br>"
                        "CO₂ per GDP: %{customdata[3]:,.2f} t CO₂/kUSD/yr"
                        "<extra></extra>"
                    ),
                )
            )
            map_trace_meta.append(("selected_fill", map_selected_df.reset_index(drop=True)))

        if not map_marker_df.empty:
            map_fig.add_trace(
                go.Scattergeo(
                    lon=map_marker_df["lon"],
                    lat=map_marker_df["lat"],
                    mode="markers",
                    marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=0)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            map_trace_meta.append(("markers_all", map_marker_df.reset_index(drop=True)))

        if not map_filter_marker_df.empty:
            map_fig.add_trace(
                go.Scattergeo(
                    lon=map_filter_marker_df["lon"],
                    lat=map_filter_marker_df["lat"],
                    mode="markers",
                    marker=dict(
                        size=7,
                        color="rgba(255,255,255,0.92)",
                        line=dict(color=ACCENT, width=1.6),
                    ),
                    customdata=np.stack([map_filter_marker_df["country_code"], map_filter_marker_df["country"]], axis=-1),
                    hovertemplate="<b>%{customdata[1]}</b><br>Country filter selection<extra></extra>",
                    showlegend=False,
                )
            )
            map_trace_meta.append(("filter_markers", map_filter_marker_df.reset_index(drop=True)))

        if not map_selected_marker_df.empty:
            map_fig.add_trace(
                go.Scattergeo(
                    lon=map_selected_marker_df["lon"],
                    lat=map_selected_marker_df["lat"],
                    mode="markers",
                    marker=dict(size=10, color=HIGHLIGHT, line=dict(color="white", width=1.4)),
                    customdata=np.stack([map_selected_marker_df["country_code"], map_selected_marker_df["country"]], axis=-1),
                    hovertemplate="<b>%{customdata[1]}</b><br>Selected country<extra></extra>",
                    showlegend=False,
                )
            )
            map_trace_meta.append(("selected_markers", map_selected_marker_df.reset_index(drop=True)))

        map_fig.update_layout(
            template="plotly_white",
            title=chart_title(f"Fossil CO₂ per capita by country ({selected_year})", 19),
            height=470,
            margin=dict(l=0, r=0, t=72, b=0),
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type="equirectangular",
                bgcolor="rgba(0,0,0,0)",
                showcountries=True,
                countrycolor="white",
            ),
        )

    map_points = plotly_events(
        map_fig,
        click_event=True,
        select_event=False,
        hover_event=False,
        key=f"map_chart_{revision}",
        override_height=470,
        override_width="100%",
        config=GRAPH_CONFIG_MAP,
    )

    st.caption(
        "The click-enabled map uses a custom Plotly component, so Streamlit’s native fullscreen button is not available on this map. Use the large map view below when you want a native fullscreen-style view."
    )
    with st.expander("Open large map view with native fullscreen"):
        large_map_fig = go.Figure(map_fig)
        large_map_fig.update_layout(height=700, margin=dict(l=0, r=0, t=72, b=0))
        st.plotly_chart(large_map_fig, use_container_width=True, config=GRAPH_CONFIG_VIEW_ONLY)

    # ----------------------------
    # Bar and Scatter
    # ----------------------------
    left, right = st.columns(2)
    bar_df = dff.dropna(subset=["total_mt_co2"]).copy().sort_values("total_mt_co2", ascending=False)

    if bar_df.empty:
        bar_fig = empty_figure("Largest fossil CO₂ emitters", "Mt CO₂/yr", "Country", 330)
        top_df = pd.DataFrame(columns=["country_code"])
    else:
        top_df = bar_df.head(st.session_state["top_n"]).copy()
        extra_codes = [code for code in selected_country_codes if code not in set(top_df["country_code"])]
        if extra_codes:
            extra_rows = bar_df[bar_df["country_code"].isin(extra_codes)].copy()
            top_df = pd.concat([top_df, extra_rows], ignore_index=True)
            top_df = top_df.sort_values("total_mt_co2", ascending=False).drop_duplicates(subset=["country_code"]).copy()

        top_df = top_df.sort_values("total_mt_co2", ascending=True)
        colors = [HIGHLIGHT if code in set(selected_country_codes) else ACCENT for code in top_df["country_code"]]

        bar_fig = go.Figure()
        bar_fig.add_trace(
            go.Bar(
                x=top_df["total_mt_co2"],
                y=top_df["country"],
                orientation="h",
                marker=dict(color=colors),
                customdata=np.stack([top_df["country_code"]], axis=-1),
                hovertemplate="<b>%{y}</b><br>Total emissions: %{x:,.1f} Mt CO₂/yr<extra></extra>",
                showlegend=False,
            )
        )
        bar_fig.update_layout(
            template="plotly_white",
            title=chart_title(
                bar_title_text(
                    filtered_count=dff["country"].nunique(),
                    requested_top_n=st.session_state["top_n"],
                    actual_shown_count=top_df.shape[0],
                    selected_year=selected_year,
                    selected_outside_topn=selected_outside_topn,
                    selected_count=len(selected_country_codes),
                ),
                18,
            ),
            xaxis_title="Total fossil CO₂ emissions (Mt CO₂/yr)",
            yaxis_title="Country",
            height=330,
            margin=dict(l=10, r=14, t=72, b=18),
        )
        bar_fig.update_xaxes(tickformat=",d")

    with left:
        bar_event = st.plotly_chart(
            bar_fig,
            use_container_width=True,
            config=GRAPH_CONFIG_SELECT,
            key=f"bar_select_chart_{revision}",
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
        )

    scatter_df = dff.dropna(subset=["co2_per_gdp_t_per_kusd", "co2_per_capita_t", "total_mt_co2"]).copy()

    if scatter_df.empty:
        scatter_fig = empty_figure(
            "CO₂ intensity vs per-capita emissions",
            "CO₂ per GDP (t CO₂/kUSD/yr)",
            "CO₂ per capita (t CO₂/cap/yr)",
            350,
        )
    else:
        selected_set = set(selected_country_codes)
        scatter_base_df = scatter_df[~scatter_df["country_code"].isin(selected_set)].copy()
        scatter_selected_df = scatter_df[scatter_df["country_code"].isin(selected_set)].copy()

        scatter_fig = go.Figure()
        if not scatter_base_df.empty:
            scatter_fig.add_trace(
                go.Scatter(
                    x=scatter_base_df["co2_per_gdp_t_per_kusd"],
                    y=scatter_base_df["co2_per_capita_t"],
                    mode="markers",
                    marker=dict(
                        size=get_bubble_sizes(scatter_base_df["total_mt_co2"], min_size=8, max_size=32),
                        color=ACCENT,
                        opacity=0.60 if selected_country_codes else 0.80,
                        line=dict(color="white", width=0.5),
                    ),
                    customdata=np.stack(
                        [scatter_base_df["country_code"], scatter_base_df["country"], scatter_base_df["total_mt_co2"]],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "CO₂ per GDP: %{x:,.2f} t CO₂/kUSD/yr<br>"
                        "CO₂ per capita: %{y:,.2f} t CO₂/cap/yr<br>"
                        "Total emissions: %{customdata[2]:,.1f} Mt CO₂/yr"
                        "<extra></extra>"
                    ),
                    name="Comparison set",
                )
            )

        if not scatter_selected_df.empty:
            scatter_fig.add_trace(
                go.Scatter(
                    x=scatter_selected_df["co2_per_gdp_t_per_kusd"],
                    y=scatter_selected_df["co2_per_capita_t"],
                    mode="markers",
                    marker=dict(
                        size=get_bubble_sizes(scatter_selected_df["total_mt_co2"], min_size=16, max_size=30),
                        color=HIGHLIGHT,
                        opacity=0.96,
                        line=dict(color="white", width=1.2),
                    ),
                    customdata=np.stack(
                        [scatter_selected_df["country_code"], scatter_selected_df["country"], scatter_selected_df["total_mt_co2"]],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "CO₂ per GDP: %{x:,.2f} t CO₂/kUSD/yr<br>"
                        "CO₂ per capita: %{y:,.2f} t CO₂/cap/yr<br>"
                        "Total emissions: %{customdata[2]:,.1f} Mt CO₂/yr"
                        "<extra></extra>"
                    ),
                    name="Selected",
                )
            )

        scatter_fig.update_layout(
            template="plotly_white",
            title=chart_title(f"CO₂ intensity vs per-capita emissions ({selected_year})", 18),
            xaxis_title="CO₂ per GDP (t CO₂/kUSD/yr)",
            yaxis_title="CO₂ per capita (t CO₂/cap/yr)",
            hovermode="closest",
            height=350,
            margin=dict(l=10, r=118, t=78, b=64),
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="rgba(226,232,240,0.9)",
                borderwidth=1,
                title_text="",
            ),
        )
        scatter_fig.update_xaxes(tickformat=".2f", title_standoff=10)
        scatter_fig.update_yaxes(tickformat=".0f", title_standoff=12)

    with right:
        st.caption("x = CO₂ per GDP, y = CO₂ per capita, bubble size = total emissions")
        scatter_event = st.plotly_chart(
            scatter_fig,
            use_container_width=True,
            config=GRAPH_CONFIG_SELECT,
            key=f"scatter_select_chart_{revision}",
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
        )

    # ----------------------------
    # Sector chart
    # ----------------------------
    sector_year_df = sector_long[sector_long["year"] == selected_year].copy()
    _, sector_dff = build_sector_comparison_view(
        sector_year_df=sector_year_df,
        selected_countries=st.session_state["countries"],
        selected_country_codes=selected_country_codes,
    )

    shown_bar_codes = top_df["country_code"].tolist() if not top_df.empty else []
    filter_scope_codes = country_filter_codes
    if filter_scope_codes or selected_country_codes:
        sector_scope_codes = unique_preserve_order(filter_scope_codes + selected_country_codes)
        has_filter_focus = len(filter_scope_codes) > 0
    else:
        sector_scope_codes = shown_bar_codes
        has_filter_focus = False

    comparison_country_df = (
        dff[dff["country_code"].isin(sector_scope_codes)][["country_code", "country", "total_mt_co2"]]
        .dropna(subset=["country_code", "country"])
        .drop_duplicates(subset=["country_code"])
        .copy()
    )
    if not comparison_country_df.empty:
        comparison_country_df["is_selected"] = comparison_country_df["country_code"].isin(selected_country_codes)
        comparison_country_df = comparison_country_df.sort_values(
            ["is_selected", "total_mt_co2"], ascending=[False, False]
        )
    sector_plot_df = sector_dff[sector_dff["country_code"].isin(sector_scope_codes)].copy()

    sector_selected_names = (
        comparison_country_df.set_index("country_code").reindex(selected_country_codes)["country"].dropna().tolist()
        if selected_country_codes else []
    )
    sector_title = sector_title_text(
        selected_year=selected_year,
        selected_names=sector_selected_names,
        has_filter_focus=has_filter_focus,
        shown_count=len(sector_scope_codes),
    )

    if comparison_country_df.empty or "sector" not in sector_plot_df.columns or "sector_mt_co2" not in sector_plot_df.columns:
        sector_fig = empty_figure(sector_title, "Country", "Mt CO₂/yr", 430)
    else:
        sector_order = (
            sector_plot_df.groupby("sector", as_index=False)["sector_mt_co2"]
            .sum()
            .sort_values("sector_mt_co2", ascending=False)
        )
        top_sectors = sector_order["sector"].head(6).tolist()
        sector_plot_df["sector_group"] = np.where(
            sector_plot_df["sector"].isin(top_sectors),
            sector_plot_df["sector"],
            "Other sectors",
        )
        sector_agg = sector_plot_df.groupby(["country", "country_code", "sector_group"], as_index=False)["sector_mt_co2"].sum()

        display_country_df = comparison_country_df[["country_code", "country", "is_selected"]].copy()
        display_country_df["display_country"] = np.where(
            display_country_df["is_selected"],
            "★ " + display_country_df["country"],
            display_country_df["country"],
        )
        code_to_display = dict(zip(display_country_df["country_code"], display_country_df["display_country"]))
        sector_agg["display_country"] = sector_agg["country_code"].map(code_to_display).fillna(sector_agg["country"])

        country_totals = (
            sector_agg.groupby(["country_code", "display_country"], as_index=False)["sector_mt_co2"]
            .sum()
            .rename(columns={"sector_mt_co2": "sector_total_mt_co2"})
        )
        display_country_df = display_country_df.merge(country_totals, on=["country_code", "display_country"], how="left")
        display_country_df["sector_total_mt_co2"] = display_country_df["sector_total_mt_co2"].fillna(0.0)
        display_country_df = display_country_df.sort_values(
            ["sector_total_mt_co2", "country"], ascending=[False, True]
        )
        country_order = display_country_df["display_country"].tolist()

        sector_names_present = (
            sector_agg.groupby("sector_group", as_index=False)["sector_mt_co2"]
            .sum()
            .sort_values("sector_mt_co2", ascending=False)["sector_group"]
            .tolist()
        )
        ordered_sector_names = [s for s in PREFERRED_SECTOR_ORDER if s in sector_names_present]
        ordered_sector_names += [s for s in sector_names_present if s not in ordered_sector_names]
        palette = choose_sector_palette(ordered_sector_names)

        sector_fig = go.Figure()
        for sector_name in ordered_sector_names:
            df_s = (
                sector_agg[sector_agg["sector_group"] == sector_name][["display_country", "sector_mt_co2"]]
                .drop_duplicates(subset=["display_country"])
                .set_index("display_country")
                .reindex(country_order, fill_value=0.0)
                .reset_index()
            )
            sector_fig.add_trace(
                go.Bar(
                    x=df_s["display_country"],
                    y=df_s["sector_mt_co2"],
                    name=sector_name,
                    marker=dict(color=palette.get(sector_name, "#999999"), line=dict(color="white", width=0.6)),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        f"Sector: {sector_name}<br>"
                        "Emissions: %{y:,.1f} Mt CO₂/yr"
                        "<extra></extra>"
                    ),
                )
            )

        sector_fig.update_layout(
            template="plotly_white",
            title=chart_title(sector_title, 18),
            xaxis_title="Country",
            yaxis_title="Sector emissions (Mt CO₂/yr)",
            barmode="stack",
            height=430,
            margin=dict(l=10, r=170, t=72, b=40),
            legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02, title_text="Sector"),
        )
        sector_fig.update_yaxes(tickformat=",d")
        sector_fig.update_xaxes(tickangle=25)

    if filter_scope_codes and selected_country_codes:
        sector_caption = "The sector chart shows the current focus in descending order: countries from the filter plus any countries selected from the map, bar, or scatter. Selected countries are marked with ★ on the x-axis."
    elif filter_scope_codes:
        sector_caption = "The sector chart follows the current country filter. Add map, bar, or scatter selections to compare extra countries without losing the filtered set."
    elif selected_country_codes:
        sector_caption = "The sector chart is in focus mode for the countries selected from the map, bar, or scatter. Selected countries are marked with ★ on the x-axis."
    else:
        sector_caption = "The sector chart defaults to the largest-emitter comparison. Use the country filter or select countries from the visuals to drill into sector contributions."
    st.caption(sector_caption)
    st.plotly_chart(sector_fig, use_container_width=True, config=GRAPH_CONFIG_VIEW_ONLY)

# ----------------------------
# Process selections after rendering
# ----------------------------
lookups = {"map_trace_meta": map_trace_meta}

if st.session_state.pop("skip_event_processing_once", False):
    pass
else:
    changed = False
    changed = process_map_click(map_points, lookups) or changed
    changed = process_native_selection("bar", bar_event) or changed
    changed = process_native_selection("scatter", scatter_event) or changed

    if changed:
        st.rerun()
