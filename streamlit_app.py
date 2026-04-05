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
st.set_page_config(
    page_title="Worldwide CO₂ Emissions Dashboard",
    layout="wide",
)

# ----------------------------
# Paths and constants
# ----------------------------
HERE = Path(__file__).resolve().parent
COUNTRY_YEAR_FILE = HERE / "co2_country_year_merged.csv"
SECTOR_FILE = HERE / "co2_sector_country_long.csv"
CENTROIDS_FILE = HERE / "country_centroids.csv"

ACCENT = "#2563EB"        # context blue
HIGHLIGHT = "#F59E0B"     # selected focus orange
NEUTRAL = "#475569"
TEXT = "#0F172A"
BORDER = "#E2E8F0"
BG = "#F8FAFC"

GRAPH_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "scrollZoom": False,
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


def empty_figure(title: str, x_title: str = "", y_title: str = "", height: int = 320):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 20}},
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
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
    for s in sectors:
        if s in base:
            palette[s] = base[s]
        else:
            palette[s] = extras[extra_idx % len(extras)]
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


def bar_title_text(
    filtered_count: int,
    requested_top_n: int,
    actual_shown_count: int,
    selected_year: int,
    selected_outside_topn: bool,
) -> str:
    if filtered_count <= 1:
        return f"Emissions in filtered view ({selected_year})"
    if selected_outside_topn:
        return f"Top {requested_top_n} Emitters + Selected Country ({selected_year})"
    return f"Top {actual_shown_count} Emitters ({selected_year})"


def sector_title_text(
    active_country_name: Optional[str],
    selected_year: int,
    shown_count: int,
    filtered_count: int,
) -> str:
    if active_country_name:
        return f"Sector contributions for {active_country_name} ({selected_year})"
    if filtered_count <= 1:
        return f"Sector contributions in current filtered view ({selected_year})"
    return f"Sector contributions for top {shown_count} emitters ({selected_year})"


def choose_selected_textposition(
    all_df: pd.DataFrame,
    selected_x: float,
    selected_y: float,
) -> str:
    if all_df.empty:
        return "top center"

    x_med = float(all_df["co2_per_gdp_t_per_kusd"].median())
    y_med = float(all_df["co2_per_capita_t"].median())

    if selected_x >= x_med and selected_y >= y_med:
        return "bottom left"
    if selected_x >= x_med and selected_y < y_med:
        return "top left"
    if selected_x < x_med and selected_y >= y_med:
        return "bottom right"
    return "top right"


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


def update_focus_from_event(chart_name: str, points: list, lookups: dict):
    if not points:
        return False

    point = points[0]
    focus_code = None

    if chart_name == "map":
        curve = point.get("curveNumber", 0)
        idx = point.get("pointIndex", point.get("pointNumber"))

        if curve == 0 and idx is not None and 0 <= idx < len(lookups["map_df"]):
            focus_code = lookups["map_df"].iloc[idx]["country_code"]
        elif curve == 1 and idx is not None and 0 <= idx < len(lookups["map_focus_df"]):
            focus_code = lookups["map_focus_df"].iloc[idx]["country_code"]
        elif curve == 2 and idx is not None and 0 <= idx < len(lookups["map_marker_df"]):
            focus_code = lookups["map_marker_df"].iloc[idx]["country_code"]
        elif curve == 3 and idx is not None and 0 <= idx < len(lookups["map_focus_marker_df"]):
            focus_code = lookups["map_focus_marker_df"].iloc[idx]["country_code"]

    elif chart_name == "bar":
        idx = point.get("pointIndex", point.get("pointNumber"))
        if idx is not None and 0 <= idx < len(lookups["bar_df"]):
            focus_code = lookups["bar_df"].iloc[idx]["country_code"]

    elif chart_name == "scatter":
        curve = point.get("curveNumber", 0)
        idx = point.get("pointIndex", point.get("pointNumber"))
        if curve == 0 and idx is not None and 0 <= idx < len(lookups["scatter_base_df"]):
            focus_code = lookups["scatter_base_df"].iloc[idx]["country_code"]
        elif curve == 1 and idx is not None and 0 <= idx < len(lookups["scatter_focus_df"]):
            focus_code = lookups["scatter_focus_df"].iloc[idx]["country_code"]

    if focus_code is None:
        return False

    current_focus = st.session_state.get("focus_country_code")
    st.session_state["focus_country_code"] = None if current_focus == focus_code else focus_code
    st.session_state["event_nonce"] = st.session_state.get("event_nonce", 0) + 1
    return True


# ----------------------------
# Load data
# ----------------------------
country_year, sector_long, centroids = load_data()
YEARS = sorted(country_year["year"].dropna().unique().tolist())
COUNTRIES = sorted(country_year["country"].dropna().unique().tolist())

# ----------------------------
# Session state
# ----------------------------
if "year" not in st.session_state:
    st.session_state["year"] = max(YEARS)
if "countries" not in st.session_state:
    st.session_state["countries"] = []
if "top_n" not in st.session_state:
    st.session_state["top_n"] = 10
if "focus_country_code" not in st.session_state:
    st.session_state["focus_country_code"] = None
if "event_nonce" not in st.session_state:
    st.session_state["event_nonce"] = 0


def reset_all():
    st.session_state["year"] = max(YEARS)
    st.session_state["countries"] = []
    st.session_state["top_n"] = 10
    st.session_state["focus_country_code"] = None
    st.session_state["event_nonce"] = st.session_state.get("event_nonce", 0) + 1


# ----------------------------
# Page styles
# ----------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 1.25rem; max-width: 1500px;}
    .main-title {font-size: 2rem; font-weight: 700; color: #0F172A; margin-bottom: 0.1rem;}
    .sub-title {color: #475569; font-size: 0.95rem; margin-bottom: 0.1rem;}
    .tip {color: #64748B; font-size: 0.8rem; margin-bottom: 1rem;}
    .card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 0.6rem 0.9rem 0.2rem 0.9rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        margin-bottom: 0.9rem;
    }
    .status-box {border-top: 1px solid #E2E8F0; margin-top: 1rem; padding-top: 1rem; color: #475569; font-size: 0.92rem; white-space: pre-line;}
    .small-note {color: #64748B; font-size: 0.78rem; margin-top: 0.75rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Worldwide CO₂ Emissions Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Explore which countries emit the most, how carbon-intensive they are, and which sectors explain those differences.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="tip">Tip: click a country on the map, bar chart, or bubble chart to focus it across the dashboard. Click the same country again to clear. Use Reset filters and focus to reset everything.</div>',
    unsafe_allow_html=True,
)

# ----------------------------
# Layout
# ----------------------------
sidebar_col, main_col = st.columns([0.24, 0.76], vertical_alignment="top")

with sidebar_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Filters")

    st.selectbox("Year", YEARS, key="year")
    st.multiselect("Country", COUNTRIES, key="countries", placeholder="All countries")
    st.slider("Top emitters to show", min_value=5, max_value=20, step=1, key="top_n")

    if st.button("Reset filters and focus", use_container_width=True):
        reset_all()
        st.rerun()

selected_year = st.session_state["year"]

# Full-year world context for map
year_df = country_year[country_year["year"] == selected_year].copy()
year_df = year_df.dropna(subset=["country_code", "country"]).copy()

# Sidebar-filtered dataframe for comparison visuals
dff = year_df.copy()
if st.session_state["countries"]:
    dff = dff[dff["country"].isin(st.session_state["countries"])].copy()

available_year_codes = set(year_df["country_code"].tolist())
active_country_code = (
    st.session_state["focus_country_code"]
    if st.session_state["focus_country_code"] in available_year_codes
    else None
)

active_country_name = None
if active_country_code:
    match = year_df.loc[year_df["country_code"] == active_country_code, "country"]
    if not match.empty:
        active_country_name = match.iloc[0]

# Ranking within filtered comparison view
bar_rank_df = (
    dff.dropna(subset=["total_mt_co2"])
    .copy()
    .sort_values("total_mt_co2", ascending=False)
    .reset_index(drop=True)
)
bar_rank_df["rank_total_emissions"] = np.arange(1, len(bar_rank_df) + 1)

selected_rank = None
selected_outside_topn = False
if active_country_code and not bar_rank_df.empty:
    rank_match = bar_rank_df.loc[bar_rank_df["country_code"] == active_country_code, "rank_total_emissions"]
    if not rank_match.empty:
        selected_rank = int(rank_match.iloc[0])
        selected_outside_topn = selected_rank > st.session_state["top_n"]

scatter_df_for_status = dff.dropna(subset=["co2_per_gdp_t_per_kusd", "co2_per_capita_t", "total_mt_co2"]).copy()
scatter_available = scatter_df_for_status["country"].nunique() if not scatter_df_for_status.empty else 0

rank_line = (
    f"Clicked country rank by total emissions: {selected_rank}"
    if selected_rank else
    "Clicked country rank by total emissions: None"
)
compare_line = (
    f"Clicked country is outside the top {st.session_state['top_n']} emitters and has been added to the bar chart for comparison."
    if selected_outside_topn else
    f"Clicked country is within the top {st.session_state['top_n']} emitters."
    if selected_rank else
    "No country clicked."
)

with sidebar_col:
    st.markdown(
        f"""
        <div class="status-box">
        Year: {selected_year}
        <br>Countries in filtered comparison view: {dff['country'].nunique():,}
        <br>Scatter countries with complete intensity data: {scatter_available:,}
        <br>Clicked country: {active_country_name if active_country_name else 'None'}
        <br>{rank_line}
        <br>{compare_line}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="small-note">
        Color guide: the blue scale next to the map belongs only to the choropleth and represents CO₂ per capita (t CO₂/cap/yr). In the other charts, blue shows all filtered countries, orange highlights the clicked country, and sector colors identify emission sources.
        <br><br>
        Bubble size in the scatter shows total emissions (Mt CO₂/yr). The scatter excludes countries with missing GDP- or per-capita intensity values for the selected year. If a clicked country is outside the top-N emitters, it is still appended to the bar chart for comparison.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with main_col:
    # ----------------------------
    # Map
    # ----------------------------
    map_df = year_df.dropna(subset=["co2_per_capita_t"]).copy()

    focus_codes = set()
    if st.session_state["countries"]:
        focus_codes.update(
            year_df.loc[
                year_df["country"].isin(st.session_state["countries"]), "country_code"
            ].dropna().astype(str).str.upper().tolist()
        )
    if active_country_code:
        focus_codes.add(active_country_code)

    map_focus_df = map_df[map_df["country_code"].isin(focus_codes)].copy()

    map_marker_df = map_df[["country_code", "country"]].drop_duplicates().copy()
    map_marker_df = map_marker_df.merge(centroids, on="country_code", how="left").dropna(subset=["lat", "lon"])

    map_focus_marker_df = map_focus_df[["country_code", "country"]].drop_duplicates().copy()
    map_focus_marker_df = map_focus_marker_df.merge(centroids, on="country_code", how="left").dropna(subset=["lat", "lon"])

    if map_df.empty:
        map_fig = empty_figure("CO₂ per capita by country", height=470)
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

        if not map_focus_df.empty:
            map_fig.add_trace(
                go.Choropleth(
                    locations=map_focus_df["country_code"],
                    z=np.ones(len(map_focus_df)),
                    locationmode="ISO-3",
                    colorscale=[[0, HIGHLIGHT], [1, HIGHLIGHT]],
                    showscale=False,
                    marker_line_color=HIGHLIGHT,
                    marker_line_width=2.5,
                    customdata=np.stack(
                        [
                            map_focus_df["country_code"],
                            map_focus_df["country"],
                            map_focus_df["total_mt_co2"].fillna(np.nan),
                            map_focus_df["co2_per_gdp_t_per_kusd"].fillna(np.nan),
                            map_focus_df["co2_per_capita_t"].fillna(np.nan),
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

        if not map_focus_marker_df.empty:
            map_fig.add_trace(
                go.Scattergeo(
                    lon=map_focus_marker_df["lon"],
                    lat=map_focus_marker_df["lat"],
                    mode="markers+text" if len(map_focus_marker_df) <= 3 else "markers",
                    text=map_focus_marker_df["country"] if len(map_focus_marker_df) <= 3 else None,
                    textposition="top center",
                    marker=dict(
                        size=11,
                        color=HIGHLIGHT,
                        line=dict(color="white", width=1.6),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        map_fig.update_layout(
            template="plotly_white",
            title={
                "text": f"CO₂ per capita by country ({selected_year})",
                "x": 0.02,
                "xanchor": "left",
                "font": {"size": 20},
            },
            height=470,
            margin=dict(l=0, r=0, t=55, b=0),
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type="equirectangular",
                bgcolor="rgba(0,0,0,0)",
                showcountries=True,
                countrycolor="white",
            ),
            clickmode="event+select",
        )

    map_points = plotly_events(
        map_fig,
        click_event=True,
        select_event=False,
        hover_event=False,
        override_height=470,
        override_width="100%",
        config=GRAPH_CONFIG,
        key=f"map_chart_{st.session_state['event_nonce']}",
    )

    # ----------------------------
    # Bar and Scatter
    # ----------------------------
    left, right = st.columns(2)

    bar_df = dff.dropna(subset=["total_mt_co2"]).copy().sort_values("total_mt_co2", ascending=False)

    if bar_df.empty:
        bar_fig = empty_figure("Top total emitters", "Mt CO₂/yr", "Country", 320)
        top_df = pd.DataFrame(columns=["country_code"])
    else:
        top_df = bar_df.head(st.session_state["top_n"]).copy()

        if active_country_code and active_country_code not in set(top_df["country_code"]):
            extra = bar_df[bar_df["country_code"] == active_country_code].head(1)
            top_df = pd.concat([top_df, extra], ignore_index=True)
            top_df = (
                top_df.sort_values("total_mt_co2", ascending=False)
                .drop_duplicates(subset=["country_code"])
                .copy()
            )

        top_df = top_df.sort_values("total_mt_co2", ascending=True)
        colors = [HIGHLIGHT if code == active_country_code else ACCENT for code in top_df["country_code"]]

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
            title={
                "text": bar_title_text(
                    filtered_count=dff["country"].nunique(),
                    requested_top_n=st.session_state["top_n"],
                    actual_shown_count=top_df.shape[0],
                    selected_year=selected_year,
                    selected_outside_topn=selected_outside_topn,
                ),
                "x": 0.02,
                "xanchor": "left",
                "font": {"size": 17},
            },
            xaxis_title="Total fossil CO₂ emissions (Mt CO₂/yr)",
            yaxis_title="Country",
            height=320,
            margin=dict(l=10, r=10, t=72, b=10),
            clickmode="event+select",
        )
        bar_fig.update_xaxes(tickformat=",d")

    with left:
        bar_points = plotly_events(
            bar_fig,
            click_event=True,
            select_event=False,
            hover_event=False,
            override_height=320,
            override_width="100%",
            config=GRAPH_CONFIG,
            key=f"bar_chart_{st.session_state['event_nonce']}",
        )

    scatter_df = dff.dropna(subset=["co2_per_gdp_t_per_kusd", "co2_per_capita_t", "total_mt_co2"]).copy()

    if scatter_df.empty:
        scatter_fig = empty_figure(
            "CO₂ intensity comparison",
            "CO₂ per GDP (t CO₂/kUSD/yr)",
            "CO₂ per capita (t CO₂/cap/yr)",
            340,
        )
        scatter_base_df = pd.DataFrame(columns=["country_code"])
        scatter_focus_df = pd.DataFrame(columns=["country_code"])
    else:
        scatter_fig = go.Figure()

        base_mask = np.ones(len(scatter_df), dtype=bool)
        if active_country_code:
            base_mask = scatter_df["country_code"] != active_country_code

        scatter_base_df = scatter_df[base_mask].copy()
        scatter_focus_df = scatter_df[scatter_df["country_code"] == active_country_code].copy() if active_country_code else pd.DataFrame()

        if not scatter_base_df.empty:
            scatter_fig.add_trace(
                go.Scatter(
                    x=scatter_base_df["co2_per_gdp_t_per_kusd"],
                    y=scatter_base_df["co2_per_capita_t"],
                    mode="markers",
                    marker=dict(
                        size=get_bubble_sizes(scatter_base_df["total_mt_co2"], min_size=8, max_size=32),
                        color=ACCENT,
                        opacity=0.55 if active_country_code else 0.75,
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
                    name="All filtered countries",
                )
            )

        if not scatter_focus_df.empty:
            sel_x = float(scatter_focus_df["co2_per_gdp_t_per_kusd"].iloc[0])
            sel_y = float(scatter_focus_df["co2_per_capita_t"].iloc[0])
            text_pos = choose_selected_textposition(scatter_df, sel_x, sel_y)

            scatter_fig.add_trace(
                go.Scatter(
                    x=scatter_focus_df["co2_per_gdp_t_per_kusd"],
                    y=scatter_focus_df["co2_per_capita_t"],
                    mode="markers+text",
                    text=scatter_focus_df["country"],
                    textposition=text_pos,
                    marker=dict(
                        size=get_bubble_sizes(scatter_focus_df["total_mt_co2"], min_size=18, max_size=30),
                        color=HIGHLIGHT,
                        opacity=0.95,
                        line=dict(color="white", width=1.2),
                    ),
                    customdata=np.stack(
                        [scatter_focus_df["country_code"], scatter_focus_df["country"], scatter_focus_df["total_mt_co2"]],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "CO₂ per GDP: %{x:,.2f} t CO₂/kUSD/yr<br>"
                        "CO₂ per capita: %{y:,.2f} t CO₂/cap/yr<br>"
                        "Total emissions: %{customdata[2]:,.1f} Mt CO₂/yr"
                        "<extra></extra>"
                    ),
                    name="Selected country",
                )
            )

        scatter_fig.update_layout(
            template="plotly_white",
            title={
                "text": f"CO₂ intensity comparison ({selected_year})"
                        f"<br><sup>x = CO₂ per GDP, y = CO₂ per capita, bubble size = total emissions</sup>",
                "x": 0.02,
                "xanchor": "left",
                "font": {"size": 18},
            },
            xaxis_title="CO₂ per GDP (t CO₂/kUSD/yr)",
            yaxis_title="CO₂ per capita (t CO₂/cap/yr)",
            height=340,
            margin=dict(l=10, r=10, t=85, b=70),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="left",
                x=0,
            ),
            clickmode="event+select",
        )
        scatter_fig.update_xaxes(tickformat=".2f")
        scatter_fig.update_yaxes(tickformat=".0f")

    with right:
        scatter_points = plotly_events(
            scatter_fig,
            click_event=True,
            select_event=False,
            hover_event=False,
            override_height=340,
            override_width="100%",
            config=GRAPH_CONFIG,
            key=f"scatter_chart_{st.session_state['event_nonce']}",
        )

    # ----------------------------
    # Sector chart
    # ----------------------------
    sector_year_df = sector_long[sector_long["year"] == selected_year].copy()

    if st.session_state["countries"]:
        sector_year_df = sector_year_df[sector_year_df["country"].isin(st.session_state["countries"])].copy()

    if active_country_code:
        sector_plot_df = sector_year_df[sector_year_df["country_code"] == active_country_code].copy()
        sector_title = sector_title_text(active_country_name, selected_year, 0, dff["country"].nunique())
    else:
        top_codes = bar_df.head(st.session_state["top_n"])["country_code"].tolist() if not bar_df.empty else []
        sector_plot_df = sector_year_df[sector_year_df["country_code"].isin(top_codes)].copy()
        sector_title = sector_title_text(None, selected_year, len(top_codes), dff["country"].nunique())

    if sector_plot_df.empty or "sector" not in sector_plot_df.columns or "sector_mt_co2" not in sector_plot_df.columns:
        sector_fig = empty_figure(sector_title, "Country", "Mt CO₂/yr", 420)
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

        sector_agg = (
            sector_plot_df.groupby(["country", "country_code", "sector_group"], as_index=False)["sector_mt_co2"]
            .sum()
        )

        country_order = (
            sector_agg.groupby("country", as_index=False)["sector_mt_co2"]
            .sum()
            .sort_values("sector_mt_co2", ascending=False)["country"]
            .tolist()
        )

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
            df_s = sector_agg[sector_agg["sector_group"] == sector_name].copy()
            df_s["country"] = pd.Categorical(df_s["country"], categories=country_order, ordered=True)
            df_s = df_s.sort_values("country")

            sector_fig.add_trace(
                go.Bar(
                    x=df_s["country"],
                    y=df_s["sector_mt_co2"],
                    name=sector_name,
                    marker=dict(
                        color=palette.get(sector_name, "#999999"),
                        line=dict(color="white", width=0.6),
                    ),
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
            title={
                "text": sector_title,
                "x": 0.02,
                "xanchor": "left",
                "font": {"size": 20},
            },
            xaxis_title="Country",
            yaxis_title="Sector emissions (Mt CO₂/yr)",
            barmode="stack",
            height=420,
            margin=dict(l=10, r=170, t=70, b=20),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
                title_text="Sector",
            ),
        )
        sector_fig.update_yaxes(tickformat=",d")

    st.plotly_chart(sector_fig, use_container_width=True, config=GRAPH_CONFIG)

# ----------------------------
# Process click events after rendering
# ----------------------------
lookups = {
    "map_df": map_df.reset_index(drop=True),
    "map_focus_df": map_focus_df.reset_index(drop=True),
    "map_marker_df": map_marker_df.reset_index(drop=True),
    "map_focus_marker_df": map_focus_marker_df.reset_index(drop=True),
    "bar_df": top_df.reset_index(drop=True) if 'top_df' in locals() else pd.DataFrame(),
    "scatter_base_df": scatter_base_df.reset_index(drop=True) if 'scatter_base_df' in locals() else pd.DataFrame(),
    "scatter_focus_df": scatter_focus_df.reset_index(drop=True) if 'scatter_focus_df' in locals() else pd.DataFrame(),
}

changed = False
changed = update_focus_from_event("map", map_points, lookups) or changed
changed = update_focus_from_event("bar", bar_points, lookups) or changed
changed = update_focus_from_event("scatter", scatter_points, lookups) or changed

if changed:
    st.rerun()
