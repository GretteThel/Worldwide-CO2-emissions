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

ACCENT = "#2563EB"
HIGHLIGHT = "#F59E0B"
NEUTRAL = "#475569"
TEXT = "#0F172A"
BORDER = "#E2E8F0"
BG = "#F8FAFC"

GRAPH_CONFIG = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "scrollZoom": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "autoScale2d",
        "toggleSpikelines",
    ],
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
        "text": text,
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
        margin=dict(l=10, r=16, t=92, b=20),
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


def bar_title_text(
    filtered_count: int,
    requested_top_n: int,
    actual_shown_count: int,
    selected_year: int,
    selected_outside_topn: bool,
    selected_count: int,
) -> str:
    if filtered_count <= 1 and selected_count == 0:
        return f"Emissions in filtered view ({selected_year})"
    if selected_outside_topn:
        suffix = "Countries" if selected_count > 1 else "Country"
        return f"Top {requested_top_n} Emitters + Selected {suffix} ({selected_year})"
    return f"Top {actual_shown_count} Emitters ({selected_year})"


def sector_title_text(
    selected_country_names: list[str],
    selected_year: int,
    shown_count: int,
    filtered_count: int,
) -> str:
    if len(selected_country_names) == 1:
        return f"Sector contributions for {selected_country_names[0]} ({selected_year})"
    if len(selected_country_names) > 1:
        return f"Sector contributions for selected countries ({selected_year})"
    if filtered_count <= 1:
        return f"Sector contributions in current filtered view ({selected_year})"
    return f"Sector contributions for top {shown_count} emitters ({selected_year})"


def choose_selected_textposition(
    all_df: pd.DataFrame,
    selected_x: float,
    selected_y: float,
) -> str:
    if all_df.empty:
        return "middle left"

    x_med = float(all_df["co2_per_gdp_t_per_kusd"].median())
    y_q25 = float(all_df["co2_per_capita_t"].quantile(0.25))
    y_q75 = float(all_df["co2_per_capita_t"].quantile(0.75))

    if selected_y >= y_q75:
        return "bottom left" if selected_x >= x_med else "bottom right"
    if selected_y <= y_q25:
        return "top left" if selected_x >= x_med else "top right"
    return "middle left" if selected_x >= x_med else "middle right"


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


def extract_country_code_from_event(chart_name: str, points: list, lookups: dict) -> Optional[str]:
    if not points:
        return None

    point = points[0]
    country_code = None

    if chart_name == "map":
        curve = point.get("curveNumber", 0)
        idx = point.get("pointIndex", point.get("pointNumber"))

        if curve == 0 and idx is not None and 0 <= idx < len(lookups["map_df"]):
            country_code = lookups["map_df"].iloc[idx]["country_code"]
        elif curve == 1 and idx is not None and 0 <= idx < len(lookups["map_highlight_df"]):
            country_code = lookups["map_highlight_df"].iloc[idx]["country_code"]
        elif curve == 2 and idx is not None and 0 <= idx < len(lookups["map_marker_df"]):
            country_code = lookups["map_marker_df"].iloc[idx]["country_code"]
        elif curve == 3 and idx is not None and 0 <= idx < len(lookups["map_highlight_marker_df"]):
            country_code = lookups["map_highlight_marker_df"].iloc[idx]["country_code"]

    elif chart_name == "bar":
        idx = point.get("pointIndex", point.get("pointNumber"))
        if idx is not None and 0 <= idx < len(lookups["bar_df"]):
            country_code = lookups["bar_df"].iloc[idx]["country_code"]

    elif chart_name == "scatter":
        curve = point.get("curveNumber", 0)
        idx = point.get("pointIndex", point.get("pointNumber"))
        if curve == 0 and idx is not None and 0 <= idx < len(lookups["scatter_base_df"]):
            country_code = lookups["scatter_base_df"].iloc[idx]["country_code"]
        elif curve == 1 and idx is not None and 0 <= idx < len(lookups["scatter_selected_df"]):
            country_code = lookups["scatter_selected_df"].iloc[idx]["country_code"]

    return country_code


def toggle_selected_country(country_code: Optional[str]) -> bool:
    if not country_code:
        return False

    selected_codes = list(st.session_state.get("selected_country_codes", []))
    if country_code in selected_codes:
        selected_codes = [code for code in selected_codes if code != country_code]
    else:
        selected_codes.append(country_code)

    st.session_state["selected_country_codes"] = selected_codes
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
if "selected_country_codes" not in st.session_state:
    st.session_state["selected_country_codes"] = []
if "event_nonce" not in st.session_state:
    st.session_state["event_nonce"] = 0


def reset_all():
    st.session_state["year"] = max(YEARS)
    st.session_state["countries"] = []
    st.session_state["top_n"] = 10
    st.session_state["selected_country_codes"] = []
    st.session_state["event_nonce"] = st.session_state.get("event_nonce", 0) + 1


# ----------------------------
# Page styles
# ----------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.15rem; padding-bottom: 1.6rem; max-width: 1500px;}
    .main-title {font-size: 2.15rem; line-height: 1.22; font-weight: 700; color: #0F172A; margin: 0 0 0.2rem 0;}
    .sub-title {color: #475569; font-size: 0.97rem; line-height: 1.45; margin-bottom: 0.15rem;}
    .tip {color: #64748B; font-size: 0.82rem; line-height: 1.45; margin-bottom: 1.15rem;}
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
    '<div class="tip">Tip: click countries on the map, bar chart, or bubble chart to build a multi-country selection across the dashboard. Click a selected country again to remove it. Use Reset filters and focus to reset everything.</div>',
    unsafe_allow_html=True,
)


def emitters_marks_html(selected_value: int) -> str:
    marks = [5, 10, 15, 20]
    chunks = []
    for value in marks:
        color = TEXT if value == selected_value else "#94A3B8"
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
sidebar_col, main_col = st.columns([0.24, 0.76], vertical_alignment="top")

with sidebar_col:
    st.markdown("## Filters")
    st.selectbox("Year", YEARS, key="year")
    st.multiselect("Country", COUNTRIES, key="countries", placeholder="All countries")
    st.slider("Top emitters to show", min_value=5, max_value=20, step=1, key="top_n")
    st.markdown(emitters_marks_html(st.session_state["top_n"]), unsafe_allow_html=True)
    if st.button("Reset filters and focus", use_container_width=True):
        reset_all()
        st.rerun()

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
selected_outside_filter = any(code not in set(filtered_dff["country_code"]) for code in selected_country_codes)

scatter_df_for_status = dff.dropna(subset=["co2_per_gdp_t_per_kusd", "co2_per_capita_t", "total_mt_co2"]).copy()
scatter_available = scatter_df_for_status["country"].nunique() if not scatter_df_for_status.empty else 0

selected_names_text = ", ".join(selected_country_names) if selected_country_names else "None"
latest_rank_line = (
    f"Latest clicked country rank by total emissions: {selected_rank}"
    if selected_rank is not None
    else "Latest clicked country rank by total emissions: None"
)
selection_note = (
    "Selected countries outside the country filter have been added to the comparison view."
    if selected_outside_filter
    else "No selected country is outside the country filter."
    if selected_country_codes
    else "No country clicked."
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
        Color guide: the blue scale next to the map belongs only to the choropleth and represents CO₂ per capita (t CO₂/cap/yr). In the other charts, blue shows the comparison view and orange highlights the clicked countries. Sector colors identify emission sources.
        <br><br>
        Bubble size in the scatter shows total emissions (Mt CO₂/yr). Each click toggles a country on or off, so users can build a multi-country selection from the visuals.
        </div>
        """,
        unsafe_allow_html=True,
    )

with main_col:
    # ----------------------------
    # Map
    # ----------------------------
    map_df = year_df.dropna(subset=["co2_per_capita_t"]).copy()

    highlight_codes = set(
        year_df.loc[year_df["country"].isin(st.session_state["countries"]), "country_code"]
        .dropna()
        .astype(str)
        .str.upper()
        .tolist()
    )
    highlight_codes.update(selected_country_codes)

    map_highlight_df = map_df[map_df["country_code"].isin(highlight_codes)].copy()

    map_marker_df = map_df[["country_code", "country"]].drop_duplicates().copy()
    map_marker_df = map_marker_df.merge(centroids, on="country_code", how="left").dropna(subset=["lat", "lon"])

    map_highlight_marker_df = map_highlight_df[["country_code", "country"]].drop_duplicates().copy()
    map_highlight_marker_df = map_highlight_marker_df.merge(centroids, on="country_code", how="left").dropna(subset=["lat", "lon"])

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

        if not map_highlight_df.empty:
            map_fig.add_trace(
                go.Choropleth(
                    locations=map_highlight_df["country_code"],
                    z=np.ones(len(map_highlight_df)),
                    locationmode="ISO-3",
                    colorscale=[[0, HIGHLIGHT], [1, HIGHLIGHT]],
                    showscale=False,
                    marker_line_color=HIGHLIGHT,
                    marker_line_width=2.5,
                    customdata=np.stack(
                        [
                            map_highlight_df["country_code"],
                            map_highlight_df["country"],
                            map_highlight_df["total_mt_co2"].fillna(np.nan),
                            map_highlight_df["co2_per_gdp_t_per_kusd"].fillna(np.nan),
                            map_highlight_df["co2_per_capita_t"].fillna(np.nan),
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

        if not map_highlight_marker_df.empty:
            show_text = len(map_highlight_marker_df) <= 3
            map_fig.add_trace(
                go.Scattergeo(
                    lon=map_highlight_marker_df["lon"],
                    lat=map_highlight_marker_df["lat"],
                    mode="markers+text" if show_text else "markers",
                    text=map_highlight_marker_df["country"] if show_text else None,
                    textposition="top center",
                    marker=dict(size=11, color=HIGHLIGHT, line=dict(color="white", width=1.6)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        map_fig.update_layout(
            template="plotly_white",
            title=chart_title(f"CO₂ per capita by country ({selected_year})", 19),
            height=470,
            margin=dict(l=0, r=8, t=88, b=10),
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
    left, right = st.columns(2, gap="medium")

    bar_df = dff.dropna(subset=["total_mt_co2"]).copy().sort_values("total_mt_co2", ascending=False)

    if bar_df.empty:
        bar_fig = empty_figure("Top total emitters", "Mt CO₂/yr", "Country", 330)
        top_df = pd.DataFrame(columns=["country_code", "country", "total_mt_co2"])
    else:
        top_df = bar_df.head(st.session_state["top_n"]).copy()
        if selected_country_codes:
            extras = bar_df[bar_df["country_code"].isin(selected_country_codes)].copy()
            top_df = pd.concat([top_df, extras], ignore_index=True)
            top_df = top_df.drop_duplicates(subset=["country_code"]).copy()

        top_df = top_df.sort_values("total_mt_co2", ascending=True)
        colors = [HIGHLIGHT if code in selected_country_codes else ACCENT for code in top_df["country_code"]]

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
                16,
            ),
            xaxis_title="Total fossil CO₂ emissions (Mt CO₂/yr)",
            yaxis_title="Country",
            height=330,
            margin=dict(l=10, r=14, t=96, b=24),
            clickmode="event+select",
        )
        bar_fig.update_xaxes(tickformat=",d")

    with left:
        bar_points = plotly_events(
            bar_fig,
            click_event=True,
            select_event=False,
            hover_event=False,
            override_height=330,
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
            330,
        )
        scatter_base_df = pd.DataFrame(columns=["country_code"])
        scatter_selected_df = pd.DataFrame(columns=["country_code"])
    else:
        scatter_fig = go.Figure()
        selected_mask = scatter_df["country_code"].isin(selected_country_codes)
        scatter_base_df = scatter_df[~selected_mask].copy()
        scatter_selected_df = scatter_df[selected_mask].copy()

        if not scatter_base_df.empty:
            scatter_fig.add_trace(
                go.Scatter(
                    x=scatter_base_df["co2_per_gdp_t_per_kusd"],
                    y=scatter_base_df["co2_per_capita_t"],
                    mode="markers",
                    marker=dict(
                        size=get_bubble_sizes(scatter_base_df["total_mt_co2"], min_size=8, max_size=32),
                        color=ACCENT,
                        opacity=0.55 if selected_country_codes else 0.75,
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
                    name="Comparison view",
                )
            )

        if not scatter_selected_df.empty:
            show_text = len(scatter_selected_df) <= 3
            text_positions = []
            for _, row in scatter_selected_df.iterrows():
                text_positions.append(
                    choose_selected_textposition(
                        scatter_df,
                        float(row["co2_per_gdp_t_per_kusd"]),
                        float(row["co2_per_capita_t"]),
                    )
                )

            scatter_fig.add_trace(
                go.Scatter(
                    x=scatter_selected_df["co2_per_gdp_t_per_kusd"],
                    y=scatter_selected_df["co2_per_capita_t"],
                    mode="markers+text" if show_text else "markers",
                    text=scatter_selected_df["country"] if show_text else None,
                    textposition=text_positions if show_text else None,
                    marker=dict(
                        size=get_bubble_sizes(scatter_selected_df["total_mt_co2"], min_size=18, max_size=30),
                        color=HIGHLIGHT,
                        opacity=0.95,
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
                    name="Selected countries" if len(selected_country_codes) > 1 else "Selected country",
                )
            )

        scatter_fig.update_layout(
            template="plotly_white",
            title=chart_title(f"CO₂ intensity comparison ({selected_year})", 16),
            annotations=[
                dict(
                    text="x = CO₂ per GDP, y = CO₂ per capita, bubble size = total emissions",
                    x=0.02,
                    xref="paper",
                    y=1.04,
                    yref="paper",
                    xanchor="left",
                    yanchor="bottom",
                    showarrow=False,
                    font=dict(size=11, color=NEUTRAL),
                )
            ],
            xaxis_title="CO₂ per GDP (t CO₂/kUSD/yr)",
            yaxis_title="CO₂ per capita (t CO₂/cap/yr)",
            height=330,
            margin=dict(l=10, r=14, t=96, b=84),
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
            override_height=330,
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

    if selected_country_codes:
        extra_sector_rows = sector_long[
            (sector_long["year"] == selected_year)
            & (sector_long["country_code"].isin(selected_country_codes))
        ].copy()
        sector_year_df = pd.concat([sector_year_df, extra_sector_rows], ignore_index=True)
        sector_year_df = sector_year_df.drop_duplicates(subset=["country_code", "year", "sector"]).copy()
        sector_plot_df = sector_year_df[sector_year_df["country_code"].isin(selected_country_codes)].copy()
        sector_title = sector_title_text(selected_country_names, selected_year, 0, dff["country"].nunique())
    else:
        top_codes = bar_df.head(st.session_state["top_n"])["country_code"].tolist() if not bar_df.empty else []
        sector_plot_df = sector_year_df[sector_year_df["country_code"].isin(top_codes)].copy()
        sector_title = sector_title_text([], selected_year, len(top_codes), dff["country"].nunique())

    if sector_plot_df.empty or "sector" not in sector_plot_df.columns or "sector_mt_co2" not in sector_plot_df.columns:
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
            df_sector = sector_agg[sector_agg["sector_group"] == sector_name].copy()
            df_sector["country"] = pd.Categorical(df_sector["country"], categories=country_order, ordered=True)
            df_sector = df_sector.sort_values("country")
            sector_fig.add_trace(
                go.Bar(
                    x=df_sector["country"],
                    y=df_sector["sector_mt_co2"],
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
            title=chart_title(sector_title, 19),
            xaxis_title="Country",
            yaxis_title="Sector emissions (Mt CO₂/yr)",
            barmode="stack",
            height=430,
            margin=dict(l=10, r=170, t=96, b=26),
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
    "map_highlight_df": map_highlight_df.reset_index(drop=True),
    "map_marker_df": map_marker_df.reset_index(drop=True),
    "map_highlight_marker_df": map_highlight_marker_df.reset_index(drop=True),
    "bar_df": top_df.reset_index(drop=True) if 'top_df' in locals() else pd.DataFrame(),
    "scatter_base_df": scatter_base_df.reset_index(drop=True) if 'scatter_base_df' in locals() else pd.DataFrame(),
    "scatter_selected_df": scatter_selected_df.reset_index(drop=True) if 'scatter_selected_df' in locals() else pd.DataFrame(),
}

changed = False
for chart_name, points in [("map", map_points), ("bar", bar_points), ("scatter", scatter_points)]:
    clicked_code = extract_country_code_from_event(chart_name, points, lookups)
    if clicked_code:
        changed = toggle_selected_country(clicked_code) or changed

if changed:
    st.rerun()
