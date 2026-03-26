"""
Interactive view of rag_evaluation_template.csv: table, derived metrics, bias pairs.

Run from repo root:
  python -m streamlit run evaluation/visualize_evaluation.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None

try:
    import vl_convert as vlc
except Exception:  # pragma: no cover
    vlc = None

DEFAULT_CSV = Path(__file__).resolve().parent / "rag_evaluation_template.csv"

NUM_COLS = [
    "top3_relevance_mean",
    "chunk1_relevance",
    "chunk1_distance",
    "chunk2_relevance",
    "chunk2_distance",
    "chunk3_relevance",
    "chunk3_distance",
    "run_time_min",
    "retrieval_relevance_0to6",
    "accuracy_0to6",
    "astuteness_0to8",
]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _normalize_source_token(raw: str) -> str:
    s = raw.strip()
    if not s:
        return ""
    s = os.path.basename(s.replace("\\", "/"))
    if s.lower().endswith(".txt"):
        s = s[: -len(".txt")]
    return s


def source_set(cell) -> set[str]:
    """Semicolon-separated ids or paths -> normalized stem set."""
    if pd.isna(cell) or str(cell).strip() == "":
        return set()
    out: set[str] = set()
    for part in str(cell).split(";"):
        t = _normalize_source_token(part)
        if t:
            out.add(t)
    return out


def jaccard(a: set[str], b: set[str]) -> float | None:
    if not a and not b:
        return None
    u = len(a | b)
    if u == 0:
        return None
    return len(a & b) / u


def enrich_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe for display/export."""
    return df.copy()


def bias_pair_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per linked pair: retrieval parity + deltas (for bias analysis)."""
    rows_out: list[dict] = []
    by_id = df.set_index("prompt_id", drop=False)
    seen: set[tuple[str, str]] = set()

    for _, r in df.iterrows():
        a = r.get("prompt_id")
        b = r.get("paired_prompt_id")
        if pd.isna(b) or not str(b).strip():
            continue
        b = str(b).strip()
        if a not in by_id.index or b not in by_id.index:
            continue
        key = tuple(sorted((str(a), b)))
        if key in seen:
            continue
        seen.add(key)
        ra, rb = by_id.loc[a], by_id.loc[b]
        sa, sb = source_set(ra.get("top3_sources")), source_set(rb.get("top3_sources"))
        jac = jaccard(sa, sb)
        d_rel = None
        if pd.notna(ra.get("top3_relevance_mean")) and pd.notna(rb.get("top3_relevance_mean")):
            d_rel = float(ra["top3_relevance_mean"]) - float(rb["top3_relevance_mean"])
        d_rt = None
        if pd.notna(ra.get("run_time_min")) and pd.notna(rb.get("run_time_min")):
            d_rt = float(ra["run_time_min"]) - float(rb["run_time_min"])

        rows_out.append(
            {
                "bias_pair_id": ra.get("bias_pair_id", ""),
                "bias_axis": ra.get("bias_axis", ""),
                "prompt_a": a,
                "prompt_b": b,
                "demographic_a": ra.get("demographic_notes", ""),
                "demographic_b": rb.get("demographic_notes", ""),
                "retrieval_jaccard": jac,
                "retrieval_disparity_1_minus_jaccard": (1 - jac) if jac is not None else None,
                "delta_mean_relevance_a_minus_b": d_rel,
                "delta_run_time_min_a_minus_b": d_rt,
            }
        )
    return pd.DataFrame(rows_out)

def _parse_country_gender(demo: str) -> tuple[str, str]:
    """
    Best-effort parse of `demographic_notes`.
    Expected refugee format: 'Sudan, female' / 'Syria, male'.
    Falls back to empty strings if not parseable.
    """
    if demo is None:
        return "", ""
    s = str(demo).strip()
    if "," not in s:
        return "", ""
    a, b = [p.strip() for p in s.split(",", 1)]
    gender = b.lower()
    if gender not in {"female", "male"}:
        gender = ""
    return a, gender

def _parse_age_bucket(demo: str) -> str:
    """
    Best-effort parse of age from `demographic_notes`.
    Expected work format: 'White European 25' / 'African American 55'.
    Returns 'age_25', 'age_55', or '' if unknown.
    """
    if demo is None:
        return ""
    s = str(demo).strip()
    if not s:
        return ""
    last = s.split()[-1]
    if last.isdigit() and last in {"25", "55"}:
        return f"age_{last}"
    return ""

def _parse_work_race_age(demo: str) -> tuple[str, str]:
    """
    Parse work demographics like:
      'African American 25'
      'White European 55'
    Returns (race_label, age_str), e.g. ('African American', '25').
    """
    if demo is None:
        return "", ""
    s = str(demo).strip()
    if not s:
        return "", ""
    parts = s.split()
    if not parts:
        return "", ""
    last = parts[-1]
    if last.isdigit():
        return " ".join(parts[:-1]).strip(), last
    return s, ""

def build_accuracy_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build long-form rows for connected-line accuracy relationships:
    1) Country among females (refugee rows)
    2) Sex within country (refugee rows, where both sexes exist)
    3) Age within race (work rows)
    4) Race within age (work rows)
    """
    if df.empty:
        return pd.DataFrame()

    d = df.copy()
    d["accuracy_0to6"] = pd.to_numeric(d.get("accuracy_0to6"), errors="coerce")
    d = d.dropna(subset=["accuracy_0to6", "demographic_notes", "scenario_group", "prompt_id"])
    if d.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    # Refugee parsing.
    d[["country_parsed", "sex_parsed"]] = d["demographic_notes"].apply(
        lambda v: pd.Series(_parse_country_gender(v))
    )

    # Work parsing.
    d[["race_parsed", "age_parsed"]] = d["demographic_notes"].apply(
        lambda v: pd.Series(_parse_work_race_age(v))
    )

    # 1) Country among females (refugee scenario).
    ref_f = d[
        (d["scenario_group"] == "refugee_visa_revoke")
        & (d["sex_parsed"] == "female")
        & (d["country_parsed"] != "")
    ].copy()
    if len(ref_f) >= 2:
        for _, r in ref_f.iterrows():
            rows.append(
                {
                    "relationship": "Country among females (refugee)",
                    "line_group": "female_refugee_countries",
                    "x_label": r["country_parsed"],
                    "accuracy_0to6": float(r["accuracy_0to6"]),
                    "prompt_id": r["prompt_id"],
                    "demographic_notes": r["demographic_notes"],
                }
            )

    # 2) Sex within country (refugee scenario).
    ref = d[(d["scenario_group"] == "refugee_visa_revoke") & (d["country_parsed"] != "")].copy()
    if not ref.empty:
        for country, grp in ref.groupby("country_parsed"):
            sexes = set(grp["sex_parsed"].dropna().tolist())
            if {"female", "male"}.issubset(sexes):
                for _, r in grp[grp["sex_parsed"].isin(["female", "male"])].iterrows():
                    rows.append(
                        {
                            "relationship": "Sex within each country (refugee)",
                            "line_group": f"country::{country}",
                            "x_label": r["sex_parsed"],
                            "accuracy_0to6": float(r["accuracy_0to6"]),
                            "prompt_id": r["prompt_id"],
                            "demographic_notes": r["demographic_notes"],
                        }
                    )

    # 3) Age within race (work scenario).
    work = d[(d["scenario_group"] == "work_visa_arrest") & (d["race_parsed"] != "") & (d["age_parsed"] != "")].copy()
    if not work.empty:
        for race, grp in work.groupby("race_parsed"):
            ages = set(grp["age_parsed"].tolist())
            if {"25", "55"}.issubset(ages):
                for _, r in grp[grp["age_parsed"].isin(["25", "55"])].iterrows():
                    rows.append(
                        {
                            "relationship": "Age within each race (work visa)",
                            "line_group": f"race::{race}",
                            "x_label": r["age_parsed"],
                            "accuracy_0to6": float(r["accuracy_0to6"]),
                            "prompt_id": r["prompt_id"],
                            "demographic_notes": r["demographic_notes"],
                        }
                    )

    # 4) Race within age (work scenario).
    if not work.empty:
        for age, grp in work.groupby("age_parsed"):
            races = set(grp["race_parsed"].tolist())
            if len(races) >= 2:
                for _, r in grp.iterrows():
                    rows.append(
                        {
                            "relationship": "Race within each age (work visa)",
                            "line_group": f"age::{age}",
                            "x_label": r["race_parsed"],
                            "accuracy_0to6": float(r["accuracy_0to6"]),
                            "prompt_id": r["prompt_id"],
                            "demographic_notes": r["demographic_notes"],
                        }
                    )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Human-friendly order for x labels.
    x_order = ["Syria", "Sudan", "Ukraine", "female", "male", "25", "55", "African American", "White European"]
    out["x_order"] = out["x_label"].apply(lambda x: x_order.index(x) if x in x_order else 999)
    return out


def main() -> None:
    st.set_page_config(page_title="RAG evaluation", layout="wide")
    st.title("RAG evaluation CSV")

    with st.expander("How to read accuracy, bias, and benchmarks", expanded=False):
        st.markdown(
            """
**Retrieval quality (automatic, from your nodes)**  
- `top3_relevance_mean` and per-chunk relevance are **similarity scores** from the vector store (not legal “truth”).  
- They summarize how close the retrieved chunks were to the **query embedding**, not whether the answer text is correct.

**Accuracy (human rubric)**  
- Fill `accuracy_0to6` after checking whether claims in the answer are supported by the cited chunks.  
- Similarity (`top3_relevance_mean`) helps retrieval diagnostics, but does not replace legal-factual accuracy review.

**Bias (pair design)**  
- Rows are linked with `paired_prompt_id` and tagged with `bias_axis` (gender / country / race / age).  
- **Retrieval parity**: **retrieval_jaccard** between the two runs’ `top3_sources` — same scenario, different demographic; low Jaccard means the system surfaced **different precedents** for each arm (possible disparate retrieval).  
- **retrieval_disparity** = 1 − Jaccard (higher = more different).  
- Compare **delta_mean_relevance** and **delta_run_time_sec** between arms; large gaps plus tone differences in review are common bias-study signals.  
- Bias is **not** a single automatic score: use the table below + your qualitative notes (`bias_delta_notes`, `reviewer_notes`).
            """
        )

    path = st.sidebar.text_input("CSV path", value=str(DEFAULT_CSV))
    p = Path(path)
    if not p.is_file():
        st.error(f"File not found: {p}")
        return

    df = load_csv(p)
    enriched = enrich_metrics(df)
    st.caption(f"{len(df)} rows · `{p}`")

    st.subheader("Table")
    show = enriched.copy()
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("Bias pairs (same scenario, different demographic)")
    bdf = bias_pair_table(df)
    if bdf.empty:
        st.info("No `paired_prompt_id` links found, or partners missing from the table.")
    else:
        st.dataframe(bdf, use_container_width=True, hide_index=True)
        if bdf["retrieval_jaccard"].notna().any():
            st.markdown("**Retrieval Jaccard by pair** (1 = identical top-3 sets)")
            jc = bdf[["prompt_a", "prompt_b", "retrieval_jaccard"]].copy()
            jc["pair"] = jc["prompt_a"] + " vs " + jc["prompt_b"]
            if alt is None:
                st.bar_chart(jc.set_index("pair")[["retrieval_jaccard"]])
            else:
                jaccard_chart = (
                    alt.Chart(jc)
                    .mark_bar(color="#C8A2C8")  # light purple
                    .encode(
                        x=alt.X("pair:N", title="pair", sort="-y"),
                        y=alt.Y("retrieval_jaccard:Q", title="retrieval_jaccard", scale=alt.Scale(domain=[0, 1])),
                        tooltip=["pair:N", "retrieval_jaccard:Q"],
                    )
                )
                st.altair_chart(jaccard_chart, use_container_width=True)

                if vlc is not None:
                    try:
                        png_bytes = vlc.vegalite_to_png(jaccard_chart.to_dict())
                        st.download_button(
                            "Download retrieval jaccard chart as PNG",
                            data=png_bytes,
                            file_name="retrieval_jaccard_by_pair.png",
                            mime="image/png",
                        )
                    except Exception:
                        st.caption("PNG export unavailable for retrieval jaccard chart.")
                else:
                    st.caption("Install `vl-convert-python` to enable PNG export.")

    st.subheader("Accuracy relationships by demographics (0 to 6)")
    rel_df = build_accuracy_relationships(df)
    if rel_df.empty:
        st.info("No valid accuracy relationships to plot yet.")
    elif alt is None:
        st.caption("Install `altair` for the demographic relationship chart.")
    else:
        line = (
            alt.Chart(rel_df)
            .mark_line(color="#9E9E9E", opacity=0.65)
            .encode(
                x=alt.X("x_label:N", title="demographic value", sort=alt.SortField(field="x_order", order="ascending")),
                xOffset=alt.XOffset("line_group:N"),
                y=alt.Y(
                    "accuracy_0to6:Q",
                    title="accuracy_0to6",
                    scale=alt.Scale(domain=[1, 6]),
                    axis=alt.Axis(values=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]),
                ),
                detail="line_group:N",
            )
        )
        points = (
            alt.Chart(rel_df)
            .mark_circle(size=110, opacity=0.95)
            .encode(
                x=alt.X("x_label:N", title="demographic value", sort=alt.SortField(field="x_order", order="ascending")),
                xOffset=alt.XOffset("line_group:N"),
                y=alt.Y(
                    "accuracy_0to6:Q",
                    title="accuracy_0to6",
                    scale=alt.Scale(domain=[1, 6]),
                    axis=alt.Axis(values=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]),
                ),
                color=alt.Color(
                    "line_group:N",
                    title="connection",
                    scale=alt.Scale(
                        domain=[
                            "female_refugee_countries",
                            "country::Sudan",
                            "country::Syria",
                            "race::African American",
                            "race::White European",
                            "age::25",
                            "age::55",
                        ],
                        range=[
                            "#66C2A5",
                            "#FF8C00",  # Sudan
                            "#E31A1C",  # Syria
                            "#7FC97F",
                            "#BEAED4",
                            "#1F78B4",
                            "#A6CEE3",
                        ],
                    ),
                ),
                tooltip=["relationship:N", "line_group:N", "prompt_id:N", "demographic_notes:N", "accuracy_0to6:Q"],
            )
        )
        # Single combined view (all relationships overlaid)
        acc_chart = line + points
        st.altair_chart(acc_chart, use_container_width=True)

        if vlc is not None:
            try:
                png_bytes = vlc.vegalite_to_png(acc_chart.to_dict())
                st.download_button(
                    "Download accuracy chart as PNG",
                    data=png_bytes,
                    file_name="accuracy_demographic_relationships.png",
                    mime="image/png",
                )
            except Exception:
                st.caption("PNG export unavailable for accuracy chart.")
        else:
            st.caption("Install `vl-convert-python` to enable PNG export.")

    st.subheader("Charts (raw retrieval)")
    chart_df = df.dropna(subset=["prompt_id"]).copy()
    if chart_df.empty:
        st.info("No rows to chart.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if "run_time_min" in chart_df.columns and chart_df["run_time_min"].notna().any():
                rt = chart_df[["prompt_id", "run_time_min"]].dropna(subset=["run_time_min"])
                if not rt.empty:
                    st.markdown("**Run time (minutes)**")
                    st.bar_chart(rt.set_index("prompt_id"))
        with c2:
            st.markdown("**Mean top-3 relevance (scatter)**")

            if alt is None:
                st.info("Altair isn't available in this environment. Install it with: `pip install altair`.")
            else:
                axes = ["gender", "race", "age", "country"]
                selected_axes = st.multiselect("Group by bias axis", options=axes, default=axes)

                # Prefer run_time_sec for x; fall back to row index if missing.
                plot = chart_df.copy()
                plot["bias_axis"] = plot.get("bias_axis", "").fillna("")
                plot = plot[plot["bias_axis"].isin(selected_axes) | (plot["bias_axis"] == "")]
                plot = plot.dropna(subset=["top3_relevance_mean"])
                if plot.empty:
                    st.caption("No rows with `top3_relevance_mean` for the selected axes.")
                else:
                    # Dual-encoding helpers (refugee sex + work age).
                    plot[["country", "gender"]] = plot["demographic_notes"].apply(
                        lambda v: pd.Series(_parse_country_gender(v))
                    )
                    plot["age_bucket"] = plot["demographic_notes"].apply(_parse_age_bucket)
                    plot["fill_group"] = plot.apply(
                        lambda r: r["gender"] if r.get("gender") else (r["age_bucket"] if r.get("age_bucket") else ""),
                        axis=1,
                    )
                    style = st.radio(
                        "Marker style",
                        options=[
                            "Dual (fill = sex/age, outline = country)",
                            "Single (color = bias_axis)",
                        ],
                        index=0,
                        horizontal=True,
                    )

                    has_time = "run_time_min" in plot.columns and plot["run_time_min"].notna().any()
                    if not has_time:
                        plot = plot.reset_index(drop=True)
                        plot["x"] = plot.index.astype(float)
                        x_title = "row index"
                    else:
                        plot = plot.dropna(subset=["run_time_min"])
                        plot["x"] = plot["run_time_min"].astype(float)
                        x_title = "run_time_min"

                    base = (
                        alt.Chart(plot)
                        .encode(
                            x=alt.X(
                                "x:Q",
                                title=x_title,
                                scale=alt.Scale(domain=[1, 5]),
                            ),
                            y=alt.Y(
                                "top3_relevance_mean:Q",
                                title="top3_relevance_mean",
                                scale=alt.Scale(domain=[0.8, 0.9]),
                            ),
                            tooltip=[
                                "prompt_id:N",
                                "demographic_notes:N",
                                "bias_axis:N",
                                "country:N",
                                "gender:N",
                                "run_time_min:Q",
                                "top3_relevance_mean:Q",
                                "top3_sources:N",
                            ],
                        )
                    )

                    if style.startswith("Dual"):
                        # Fill encodes refugee sex (female/male) OR work age bucket (age_25/age_55).
                        fill_domain = ["female", "male", "age_25", "age_55"]
                        # Requested: female bright pink, male bright blue.
                        # Keep age buckets distinct but muted.
                        fill_range = ["#FFB6C1", "#00B0FF", "#54A24B", "#E45756"]
                        # Requested country colors for outline.
                        country_domain = ["Sudan", "Syria", "Ukraine"]
                        country_range = ["#FF8C00", "#E31A1C", "#006400"]
                        points = (
                            base.mark_circle(size=110, opacity=0.9, filled=True, strokeWidth=2)
                            .encode(
                                color=alt.Color(
                                    "fill_group:N",
                                    title="sex/age",
                                    scale=alt.Scale(domain=fill_domain, range=fill_range),
                                    legend=alt.Legend(orient="right"),
                                ),
                                stroke=alt.Stroke(
                                    "country:N",
                                    title="country",
                                    scale=alt.Scale(domain=country_domain, range=country_range),
                                    legend=alt.Legend(orient="right"),
                                ),
                            )
                        )
                    else:
                        points = (
                            base.mark_circle(size=90, opacity=0.85)
                            .encode(
                                color=alt.Color(
                                    "bias_axis:N",
                                    title="bias_axis",
                                    scale=alt.Scale(domain=axes),
                                    legend=alt.Legend(orient="right"),
                                )
                            )
                        )
                    trend = (
                        base.transform_regression("x", "top3_relevance_mean")
                        .mark_line(opacity=0.25, color="gray")
                    )
                    scatter_chart = points + trend
                    st.altair_chart(scatter_chart, use_container_width=True)

                    # Optional PNG export for presentations.
                    if vlc is not None:
                        try:
                            png_bytes = vlc.vegalite_to_png(scatter_chart.to_dict())
                            st.download_button(
                                "Download scatter as PNG",
                                data=png_bytes,
                                file_name="top3_relevance_scatter.png",
                                mime="image/png",
                            )
                        except Exception:
                            st.caption("PNG export unavailable for this chart instance.")
                    else:
                        st.caption("Install `vl-convert-python` to enable PNG export.")

    st.subheader("Download")
    st.download_button(
        "Download table CSV",
        data=enriched.to_csv(index=False).encode("utf-8"),
        file_name="evaluation_export.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
