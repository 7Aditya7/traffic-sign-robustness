"""
Dashboard: how does the traffic sign model fail, and how much worse does it get when the
images are corrupted (motion blur, spatter, noise ...)?

    python run_inference.py      (once, creates outputs/predictions.parquet)
    python dashboard.py          (then open http://127.0.0.1:8050)
    python dashboard.py --no-images    (don't load the sign photos)
"""
import argparse
import base64
import io
import math
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from sklearn.metrics import f1_score

CLEAN = "test"          # the uncorrupted test set
SHOW_IMAGES = True


# ---------------------------------------------------------------- data
parser = argparse.ArgumentParser()
parser.add_argument("--predictions", default="outputs/predictions.parquet")
parser.add_argument("--port", type=int, default=8050)
parser.add_argument("--no-images", action="store_true")
args = parser.parse_args()
SHOW_IMAGES = not args.no_images

df = pd.read_parquet(args.predictions)
required_columns = {"split", "idx", "label", "label_name", "prediction", "prediction_name", "confidence"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Prediction file is missing required columns: {sorted(missing_columns)}")
if df.duplicated(["split", "idx"]).any():
    raise ValueError("Prediction file contains duplicate (split, idx) rows")
if not df["confidence"].between(0, 1).all():
    raise ValueError("Confidence values must be between 0 and 1")
df["correct"] = df["label"] == df["prediction"]

# clean test first, then the corruptions alphabetically, train (if present) last
others = sorted(s for s in df["split"].unique() if s not in (CLEAN, "train"))
splits = [CLEAN] + others + (["train"] if "train" in df["split"].values else [])
splits = [s for s in splits if s in df["split"].values]

names = {}
for _, r in df[["label", "label_name"]].drop_duplicates().iterrows():
    names[int(r["label"])] = r["label_name"]
for _, r in df[["prediction", "prediction_name"]].drop_duplicates().iterrows():
    names.setdefault(int(r["prediction"]), r["prediction_name"])
class_ids = sorted(names)


def short(c, n=34):
    name = names[c]
    return f"{c}: {name[:n]}" + ("…" if len(name) > n else "")


# Pairing check: same row IDs and labels are necessary for image-level comparisons,
# but they do not independently prove source-image identity. Confirm that relationship
# from the dataset documentation or source metadata before making causal claims.
clean = df[df["split"] == CLEAN].sort_values("idx")
paired = {}
pairing_notes = {}
for s in splits:
    d = df[df["split"] == s].sort_values("idx")
    if s in (CLEAN, "train"):
        paired[s] = False
        continue
    unique_ids = not d["idx"].duplicated().any() and not clean["idx"].duplicated().any()
    same_ids = unique_ids and clean["idx"].tolist() == d["idx"].tolist()
    same_labels = same_ids and clean["label"].tolist() == d["label"].tolist()
    paired[s] = bool(same_labels)
    pairing_notes[s] = "row indices and labels align" if same_labels else "row indices/labels do not align"

acc = df.groupby("split")["correct"].mean()
error_by_class = df.groupby(["label", "split"])["correct"].apply(lambda x: 1 - x.mean()).unstack()
support = df.groupby(["split", "label"]).size()


def calibration(d, bins=10):
    """accuracy vs confidence per confidence bin, plus the expected calibration error"""
    b = np.minimum((d["confidence"] * bins).astype(int), bins - 1)
    t = d.assign(bin=b).groupby("bin").agg(n=("correct", "size"), conf=("confidence", "mean"), acc=("correct", "mean"))
    ece = (t["n"] / t["n"].sum() * (t["acc"] - t["conf"]).abs()).sum()
    return t, ece


def paired_rows(split):
    """Return clean/corrupt rows joined by idx after validating one-to-one alignment."""
    if not paired.get(split, False):
        return pd.DataFrame()
    c = clean[["idx", "label", "correct", "prediction"]].rename(
        columns={"label": "label_clean", "correct": "correct_clean", "prediction": "pred_clean"})
    d = df[df["split"] == split].copy()
    m = d.merge(c, on="idx", how="inner", validate="one_to_one")
    if len(m) != len(d) or not (m["label"] == m["label_clean"]).all():
        return pd.DataFrame()
    return m


def transition_counts(split):
    """Counts of clean correctness -> corrupted correctness for paired images."""
    m = paired_rows(split)
    if m.empty:
        return None
    return pd.crosstab(
        pd.Categorical(m["correct_clean"].map({True: "Correct", False: "Wrong"}),
                       categories=["Correct", "Wrong"]),
        pd.Categorical(m["correct"].map({True: "Correct", False: "Wrong"}),
                       categories=["Correct", "Wrong"]),
        dropna=False,
    ).reindex(index=["Correct", "Wrong"], columns=["Correct", "Wrong"], fill_value=0)


def transition_figure(split):
    counts = transition_counts(split)
    if counts is None:
        return go.Figure().update_layout(title="Image-level transitions unavailable: pairing check failed")
    z = counts.values
    labels = [[f"{int(v):,}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Corrupted: correct", "Corrupted: wrong"],
        y=["Clean: correct", "Clean: wrong"], text=labels, texttemplate="%{text}",
        colorscale="Blues", hovertemplate="%{y} → %{x}<br>Images: %{z:,}<extra></extra>",
        colorbar={"title": "Images"}))
    fig.update_layout(title=f"Clean → {split}: prediction outcome transitions", height=330,
                      margin={"l": 130, "r": 20, "t": 55, "b": 80}, plot_bgcolor="white")
    return fig


# ---------------------------------------------------------------- images
@lru_cache(maxsize=4)
def get_dataset(split):
    from datasets import load_dataset
    return load_dataset("tanganke/gtsrb", split=split)


@lru_cache(maxsize=2000)
def image_src(split, idx):
    if not SHOW_IMAGES:
        return None
    try:
        img = get_dataset(split)[int(idx)]["image"].convert("RGB").resize((160, 160))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def tile(split, idx, caption):
    src = image_src(split, idx)
    pic = html.Img(src=src, style={"width": "160px", "height": "160px", "objectFit": "contain", "background": "#f8fafc"}) if src else \
        html.Div("no image", style={"width": "160px", "height": "160px", "background": "#eee"})
    return html.Div([pic, html.Div(caption)], style={"width": "164px", "fontSize": "12px", "textAlign": "center"})


# ---------------------------------------------------------------- figures that don't depend on the dropdown
def overview_figure():
    colors = ["#2563eb" if s == CLEAN else "#f97316" for s in splits]
    fig = go.Figure(go.Bar(x=splits, y=acc[splits] * 100, marker_color=colors,
                           text=[f"{a:.1%}" for a in acc[splits]], textposition="outside"))
    fig.update_layout(title="Accuracy per split", yaxis={"title": "Accuracy (%)", "range": [0, 108]},
                      height=380, margin={"t": 50, "b": 40}, plot_bgcolor="white")
    return fig


def overview_table():
    rows = []
    for s in splits:
        d = df[df["split"] == s]
        wrong = d[~d["correct"]]
        _, ece = calibration(d)
        rows.append({
            "Split": s,
            "Accuracy": f"{acc[s]:.2%}",
            "Change vs clean (pp)": "-" if s == CLEAN else f"{(acc[s] - acc[CLEAN]) * 100:+.2f}",
            "Errors": len(wrong),
            "Macro F1": f"{f1_score(d['label'], d['prediction'], average='macro', zero_division=0):.3f}",
            "Avg confidence when wrong": f"{wrong['confidence'].mean():.1%}" if len(wrong) else "-",
            "ECE": f"{ece:.3f}",
        })
    return pd.DataFrame(rows)


def heatmap_figure():
    cols = [s for s in splits if s != "train"]
    table = error_by_class[cols]
    # worst classes (on average over the corrupted splits) at the top
    order = table[[c for c in cols if c != CLEAN]].mean(axis=1).sort_values(ascending=False).index \
        if len(cols) > 1 else table.index
    table = table.loc[order]
    text = [[f"{short(c, 50)}<br>{s}: {table.loc[c, s]:.1%} error<br>n={int(support.get((s, c), 0))}" for s in cols] for c in table.index]
    fig = go.Figure(go.Heatmap(z=table.values * 100, x=cols, y=[short(c, 38) for c in table.index],
                               text=text, hoverinfo="text", colorscale="Reds",
                               colorbar={"title": "Error %"}))
    fig.update_layout(title="Error rate of every sign under every corruption (worst signs on top)",
                      height=1050, margin={"l": 300, "t": 60, "r": 90}, yaxis={"autorange": "reversed", "tickfont": {"size": 11}})
    return fig


def headline():
    if not others:
        return ""
    drops = {s: (acc[CLEAN] - acc[s]) * 100 for s in others}
    worst = max(drops, key=drops.get)
    best = min(drops, key=drops.get)
    return (f"On the clean test set the model reaches {acc[CLEAN]:.2%}. The most damaging corruption is "
            f"{worst} ({drops[worst]:.1f} percentage points lost), the least damaging is {best} "
            f"({drops[best]:.1f} pp lost).")


# ---------------------------------------------------------------- confusion network
def get_edges(split):
    wrong = df[(df["split"] == split) & ~df["correct"]]
    return wrong.groupby(["label", "prediction"]).size().rename("count").reset_index()


edges_by_split = {s: get_edges(s) for s in splits}
angle = {c: 2 * math.pi * i / len(class_ids) - math.pi / 2 for i, c in enumerate(class_ids)}
pos = {c: (math.cos(angle[c]), math.sin(angle[c])) for c in class_ids}


def visible_edges(split, view, min_count, top, focus):
    if view == "extra" and split != CLEAN:
        # Count newly broken images (clean-correct -> corrupted-wrong) by confusion pair.
        m = paired_rows(split)
        if m.empty:
            return pd.DataFrame(columns=["label", "prediction", "count"])
        new_errors = m[m["correct_clean"] & ~m["correct"]]
        e = new_errors.groupby(["label", "prediction"]).size().rename("count").reset_index()
    else:
        e = edges_by_split[split].copy()
    e = e[e["count"] >= min_count]
    if focus is not None:
        e = e[(e["label"] == focus) | (e["prediction"] == focus)]
    return e.sort_values("count", ascending=False).head(int(top))


def network_figure(split, view, min_count, top, focus):
    e = visible_edges(split, view, min_count, top, focus)
    fig = go.Figure()
    max_count = max(e["count"].max(), 1) if len(e) else 1

    mid_x, mid_y, mid_data, mid_text = [], [], [], []
    for r in e.itertuples():
        s, t = int(r.label), int(r.prediction)
        (x0, y0), (x1, y1) = pos[s], pos[t]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        # shift the arrow sideways a bit so A->B and B->A don't overlap
        ox, oy = -uy * 0.02, ux * 0.02
        start = (x0 + ux * 0.07 + ox, y0 + uy * 0.07 + oy)
        end = (x1 - ux * 0.07 + ox, y1 - uy * 0.07 + oy)
        if focus is None:
            color = "rgba(100,116,139,0.6)"
        else:
            color = "rgba(234,88,12,0.85)" if s == focus else "rgba(2,132,199,0.85)"
        fig.add_annotation(x=end[0], y=end[1], ax=start[0], ay=start[1], xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowwidth=1 + 3 * r.count / max_count, arrowcolor=color)
        mid_x.append((start[0] + end[0]) / 2)
        mid_y.append((start[1] + end[1]) / 2)
        mid_data.append([s, t])
        mid_text.append(f"{names[s]}<br>mistaken for<br>{names[t]}<br><b>{int(r.count)} images</b>")
    # trace 0 = invisible markers on the arrows (so they can be hovered / clicked)
    fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode="markers", marker={"size": 14, "opacity": 0},
                             customdata=mid_data, hovertext=mid_text, hoverinfo="text"))

    # trace 1 = the 43 sign classes
    err = [error_by_class.loc[c, split] if c in error_by_class.index else 0 for c in class_ids]
    sup = [support.get((split, c), 0) for c in class_ids]
    node_text = [f"<b>{names[c]}</b><br>{int(n)} images, error rate {er:.1%}" for c, n, er in zip(class_ids, sup, err)]
    fig.add_trace(go.Scatter(
        x=[pos[c][0] for c in class_ids], y=[pos[c][1] for c in class_ids], mode="markers+text",
        text=[str(c) for c in class_ids], textfont={"size": 10}, customdata=[[c] for c in class_ids],
        hovertext=node_text, hoverinfo="text",
        marker={"size": [16 + 14 * math.sqrt(n / max(sup)) for n in sup], "color": err, "colorscale": "Reds",
                "cmin": 0, "cmax": max(0.3, max(err)), "line": {"width": 2, "color": "#334155"},
                "colorbar": {"title": "Error rate", "tickformat": ".0%", "len": 0.6}}))
    fig.update_layout(height=850, showlegend=False, margin={"l": 5, "r": 5, "t": 5, "b": 5},
                      xaxis={"visible": False, "range": [-1.25, 1.25]},
                      yaxis={"visible": False, "range": [-1.25, 1.25], "scaleanchor": "x"},
                      plot_bgcolor="white")
    return fig


def top_confusions_table(split, view, min_count, top, focus):
    e = visible_edges(split, view, min_count, top, focus)
    rows = [{"True sign": short(int(r.label), 40), "Predicted as": short(int(r.prediction), 40),
             "Images": int(r.count)} for r in e.itertuples()]
    return dash_table.DataTable(data=rows, columns=[{"name": c, "id": c} for c in ["True sign", "Predicted as", "Images"]],
                                style_cell={"textAlign": "left", "fontSize": "12px"}, style_header={"fontWeight": "bold"})


def mistake_tiles(split, mask, n=8):
    d = df[(df["split"] == split) & mask].sort_values("confidence", ascending=False).head(n)
    return html.Div([tile(split, r.idx, f"pred {r.prediction} ({r.confidence:.0%})") for r in d.itertuples()],
                    style={"display": "flex", "flexWrap": "wrap", "gap": "8px"})


def top5_examples(split, mask, n=5):
    """Compact top-5 prediction view when the inference file contains that metadata."""
    if "top5_predictions" not in df.columns or "top5_probabilities" not in df.columns:
        return html.P("Top-5 outputs are unavailable in this prediction file. Rerun the updated run_inference.py to include them.", style=note)
    d = df[(df["split"] == split) & mask].sort_values("confidence", ascending=False).head(n)
    rows = []
    for r in d.itertuples():
        ids = list(r.top5_predictions) if r.top5_predictions is not None else []
        probs = list(r.top5_probabilities) if r.top5_probabilities is not None else []
        ranked = " | ".join(f"{names.get(int(cid), str(cid))} ({float(prob):.1%})" for cid, prob in zip(ids, probs))
        rows.append({"Image idx": int(r.idx), "True class": names.get(int(r.label), str(r.label)),
                     "Prediction": names.get(int(r.prediction), str(r.prediction)),
                     "Confidence": f"{float(r.confidence):.1%}", "Top-5 alternatives": ranked})
    return dash_table.DataTable(data=rows, columns=[{"name": c, "id": c} for c in
        ["Image idx", "True class", "Prediction", "Confidence", "Top-5 alternatives"]],
        style_cell={"textAlign": "left", "fontSize": "11px", "whiteSpace": "normal", "height": "auto", "minWidth": "90px"},
        style_data={"whiteSpace": "normal", "height": "auto"}, style_header={"fontWeight": "bold"})


# ---------------------------------------------------------------- layout
card = {"background": "white", "border": "1px solid #ddd", "borderRadius": "10px", "padding": "14px", "marginBottom": "16px"}
note = {"color": "#444", "fontSize": "14px"}

app = Dash(__name__)
app.title = "GTSRB robustness dashboard"
app.layout = html.Div(style={"maxWidth": "1250px", "margin": "auto", "padding": "16px",
                             "fontFamily": "Arial, sans-serif", "background": "#f6f7f9"}, children=[
    html.H2("Traffic sign model: confusions and robustness to image corruptions"),
    html.Div("Model: kelvinandreas/vit-traffic-sign-GTSRB    Data: tanganke/gtsrb", style=note),

    # 1 ------------------------------------------------------------
    html.H3("1. Overview: how much do the corruptions hurt?"),
    html.Div([
        html.P("Corrupted splits are compared with the clean test split. Image-level transitions are shown only when row indices and labels align; this alignment should be checked against the dataset's source-image construction before interpreting it as proof of identical source images.", style=note),
        html.P(headline(), style={"fontWeight": "bold"}),
        dcc.Graph(figure=overview_figure()),
        dash_table.DataTable(data=overview_table().to_dict("records"),
                             columns=[{"name": c, "id": c} for c in overview_table().columns],
                             style_cell={"textAlign": "left", "fontSize": "13px"}, style_header={"fontWeight": "bold"}),
        html.P("ECE (expected calibration error) = how far the model's confidence is from its real accuracy; lower is better.",
               style={**note, "marginTop": "8px"}),
    ], style=card),

    # 2 ------------------------------------------------------------
    html.H3("2. Which signs break?"),
    html.Div([
        html.P("Each row is one sign, each column one split. Dark red = the model often gets that sign wrong. "
               "Look for rows that are light in the clean column but dark under a corruption: those signs are fragile.", style=note),
        dcc.Graph(figure=heatmap_figure()),
    ], style=card),

    # split picker for the rest of the page
    html.Div([html.B("Split to inspect (sections 3, 4 and 5):  "),
              dcc.Dropdown(splits, "motion_blur" if "motion_blur" in splits else (others[0] if others else CLEAN), id="split", clearable=False,
                           style={"width": "260px", "display": "inline-block", "verticalAlign": "middle"})],
             style={**card, "position": "sticky", "top": "0", "zIndex": 5}),
    html.Div(id="kpis", style={"display": "flex", "gap": "10px", "marginBottom": "16px", "flexWrap": "wrap"}),

    # 3 ------------------------------------------------------------
    html.H3("3. What does the corruption change for individual images?"),
    html.Div([html.Div(id="flipped"), dcc.Graph(id="transitions")], style=card),

    # 4 ------------------------------------------------------------
    html.H3("4. Confusion network: which signs get mixed up?"),
    html.Div([
        html.P("Each circle is a sign class (the number is the class id, bigger = more images, redder = more errors). "
               "An arrow A to B means: images of sign A were predicted as sign B. Thicker arrow = more images. "
               "Hover for names, click a circle or an arrow to see example photos. "
               "Pick a focus sign to see only its mistakes (orange = it is mistaken for others, blue = others are mistaken for it).",
               style=note),
        html.Div([
            html.Div([html.Label("Arrows to show"),
                      dcc.RadioItems([{"label": " all errors", "value": "all"},
                                      {"label": " only errors added by the corruption (vs clean)", "value": "extra"}],
                                     "all", id="view")], style={"flex": 2}),
            html.Div([html.Label("Min. images per arrow"), dcc.Slider(1, 40, 1, value=4, id="min-count", marks=None,
                                                                      tooltip={"always_visible": True})], style={"flex": 2}),
            html.Div([html.Label("Max. arrows"), dcc.Slider(10, 80, 10, value=20, id="top", marks=None,
                                                            tooltip={"always_visible": True})], style={"flex": 2}),
            html.Div([html.Label("Focus sign"), dcc.Dropdown([{"label": short(c), "value": c} for c in class_ids], None,
                                                              id="focus", placeholder="all signs")], style={"flex": 2}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "10px"}),
        html.Div([
            html.Div(dcc.Graph(id="network", config={"displayModeBar": False}), style={"flex": 3}),
            html.Div(id="details", style={"flex": 1.3, "minWidth": "300px", "maxHeight": "700px", "overflowY": "auto"}),
        ], style={"display": "flex", "gap": "12px"}),
        html.H4("Most frequent confusions with the current settings"),
        html.Div(id="pairs"),
    ], style=card),

    # 5 ------------------------------------------------------------
    html.H3("5. Can the model's confidence be trusted?"),
    html.Div([
        html.P("Left: if the model says 90% it should be right 90% of the time (points on the dashed line). "
               "Points under the line mean over-confidence. Right: how confident the model is when it is right vs when it is wrong.",
               style=note),
        html.Div([dcc.Graph(id="reliability", style={"flex": 1}), dcc.Graph(id="conf-hist", style={"flex": 1})],
                 style={"display": "flex"}),
    ], style=card),
])


# ---------------------------------------------------------------- callbacks
def kpi(title, value, sub=""):
    return html.Div([html.Div(title, style={"fontSize": "12px", "color": "#666"}),
                     html.Div(value, style={"fontSize": "22px", "fontWeight": "bold"}),
                     html.Div(sub, style={"fontSize": "12px", "color": "#666"})],
                    style={**card, "flex": 1, "minWidth": "170px", "marginBottom": 0})


@app.callback(Output("kpis", "children"), Output("flipped", "children"), Output("transitions", "figure"), Input("split", "value"))
def update_overview(split):
    d = df[df["split"] == split]
    wrong = d[~d["correct"]]
    cards = [kpi("Accuracy", f"{acc[split]:.2%}", "clean test set" if split == CLEAN else f"{(acc[split] - acc[CLEAN]) * 100:+.2f} pp vs clean"),
             kpi("Errors", f"{len(wrong):,}", f"of {len(d):,} images"),
             kpi("Avg confidence when wrong", f"{wrong['confidence'].mean():.1%}" if len(wrong) else "-")]

    if split == CLEAN:
        text = "Select a corrupted split to compare its predictions with the clean test set."
        return cards, [html.P(text, style=note)], go.Figure().update_layout(title="Select a corrupted split", height=330)
    if not paired.get(split):
        text = "Image-level comparison is unavailable: row indices and labels do not align with the clean split."
        return cards, [html.P(text, style=note)], transition_figure(split)

    m = paired_rows(split)
    if m.empty:
        text = "Image-level comparison failed its one-to-one index/label validation."
        return cards, [html.P(text, style=note)], transition_figure(split)
    broke = m[m["correct_clean"] & ~m["correct"]]
    fixed = m[~m["correct_clean"] & m["correct"]]
    right_before = m["correct_clean"].sum()
    broken_rate = len(broke) / right_before if right_before else 0.0
    cards.append(kpi("Broken by the corruption", f"{len(broke):,}",
                     f"{broken_rate:.1%} of the images that were right on clean"))
    cards.append(kpi("Fixed by the corruption", f"{len(fixed):,}", "wrong on clean, right now"))

    worst = broke.sort_values("confidence", ascending=False).head(6)
    pairs = []
    for r in worst.itertuples():
        pairs.append(html.Div([tile(CLEAN, r.idx, f"clean: {names.get(int(r.pred_clean), r.pred_clean)}"),
                               html.Div("→", style={"fontSize": "24px", "alignSelf": "center"}),
                               tile(split, r.idx, f"pred: {names.get(int(r.prediction), r.prediction)} ({r.confidence:.0%})")],
                              style={"display": "flex", "gap": "4px", "padding": "6px", "border": "1px solid #eee"}))
    body = [html.P(f"Images classified correctly on clean data but incorrectly after '{split}'. Pairing uses matching row indices and labels; confirm source-image correspondence from the dataset definition. Below are the {len(pairs)} most confident failures (left = clean, right = corrupted).", style=note),
            html.Div(pairs, style={"display": "flex", "flexWrap": "wrap", "gap": "10px"})]
    return cards, body, transition_figure(split)


@app.callback(Output("network", "figure"), Output("pairs", "children"),
              Input("split", "value"), Input("view", "value"), Input("min-count", "value"),
              Input("top", "value"), Input("focus", "value"))
def update_network(split, view, min_count, top, focus):
    return network_figure(split, view, min_count, top, focus), top_confusions_table(split, view, min_count, top, focus)


@app.callback(Output("details", "children"), Input("network", "clickData"), Input("split", "value"))
def show_details(click, split):
    if not click:
        return html.P("Click a circle or an arrow to see details and example photos here.", style=note)
    point = click["points"][0]
    d = df[df["split"] == split]

    if point["curveNumber"] == 1:                       # a sign (circle)
        c = point["customdata"][0]
        e_now = error_by_class.loc[c, split]
        e_clean = error_by_class.loc[c, CLEAN]
        out = d[(d["label"] == c) & ~d["correct"]]["prediction"].value_counts().head(5)
        inc = d[(d["prediction"] == c) & (d["label"] != c)]["label"].value_counts().head(5)
        return html.Div([
            html.H4(short(c, 60)),
            html.P(f"Error rate in {split}: {e_now:.1%}" + ("" if split == CLEAN else f"  (clean test: {e_clean:.1%})")),
            html.B("Mistaken for:"),
            html.Ul([html.Li(f"{short(int(p), 40)}: {n}×") for p, n in out.items()] or [html.Li("nothing")]),
            html.B("Other signs wrongly predicted as this one:"),
            html.Ul([html.Li(f"{short(int(p), 40)}: {n}×") for p, n in inc.items()] or [html.Li("nothing")]),
            html.B("Most confident mistakes of this sign:"),
            mistake_tiles(split, (d["label"] == c) & ~d["correct"]),
            html.H4("Top-5 model outputs for these mistakes"),
            top5_examples(split, (d["label"] == c) & ~d["correct"]),
        ])
    s, t = point["customdata"]                          # an arrow
    return html.Div([
        html.H4("Confusion"),
        html.P(f"{names[s]}  →  predicted as  {names[t]}"),
        html.P(f"{int(((d['label'] == s) & (d['prediction'] == t)).sum())} images in {split}"),
        mistake_tiles(split, (d["label"] == s) & (d["prediction"] == t)),
        html.H4("Top-5 model outputs for these examples"),
        top5_examples(split, (d["label"] == s) & (d["prediction"] == t)),
    ])


@app.callback(Output("reliability", "figure"), Output("conf-hist", "figure"), Input("split", "value"))
def update_confidence(split):
    d = df[df["split"] == split]
    t, ece = calibration(d)
    rel = go.Figure()
    rel.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "gray"}, name="perfect"))
    rel.add_trace(go.Scatter(x=t["conf"], y=t["acc"], mode="lines+markers", name="model",
                             customdata=t["n"], hovertemplate="confidence %{x:.0%}<br>accuracy %{y:.0%}<br>n=%{customdata}<extra></extra>"))
    rel.update_layout(title=f"Reliability diagram, {split} (ECE {ece:.3f})", xaxis_title="Confidence",
                      yaxis_title="Accuracy", height=380, plot_bgcolor="white")
    hist = go.Figure()
    for ok, name, color in [(True, "correct", "#16a34a"), (False, "wrong", "#dc2626")]:
        v = d[d["correct"] == ok]["confidence"]
        if len(v):
            hist.add_trace(go.Histogram(x=v, name=f"{name} (n={len(v):,})", marker_color=color, opacity=0.7,
                                        xbins={"start": 0, "end": 1, "size": 0.05}, histnorm="probability"))
    hist.update_layout(title=f"Confidence of correct vs wrong predictions, {split}", barmode="overlay",
                       xaxis_title="Confidence", yaxis_title="Share of group", height=380, plot_bgcolor="white")
    return rel, hist


if __name__ == "__main__":
    print(acc.round(4).to_string())
    app.run(debug=False, port=args.port)
