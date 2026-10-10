"""
DataCo late-delivery risk - demo app
IT3091 Machine Learning - Group 2026-AI-05

Tab 1  Findings      what the data says, and the argument for the recommendation
Tab 2  Score order   pick a real test order and see the prediction vs the outcome
Tab 3  Try your own  change the inputs and watch the probability move
Tab 4  Model         performance on the held-out test set
Tab 5  Recommendation
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import streamlit as st
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

st.set_page_config(page_title="DataCo late-delivery risk",
                   page_icon="truck", layout="wide")

# shared palette, same as the notebooks and the report
C_BLUE, C_ORANGE, C_AQUA, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8f8e88"

matplotlib.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True,
    "grid.color": "#e8e7e1", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})


# --------------------------------------------------------------------------
# Load once
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(ROOT / "model.joblib")


@st.cache_data
def load_data():
    manifest = json.loads((REPO / "artifacts" / "manifest.json").read_text())
    train = pd.read_parquet(REPO / "artifacts" / "train.parquet")
    test = pd.read_parquet(REPO / "artifacts" / "test.parquet")
    return manifest, train, test


try:
    model = load_model()
    manifest, train, test = load_data()
except FileNotFoundError as e:
    st.error(f"Could not load the model or data: {e}")
    st.info("Run both notebooks, then `python app/build_model.py`.")
    st.stop()

TARGET = manifest["target"]
FEATURES = manifest["categorical_features"] + manifest["numeric_features"]
DATE_COL = manifest["metadata_columns"][1]
MODE_ORDER = ["First Class", "Second Class", "Same Day", "Standard Class"]


@st.cache_data
def score_test():
    return model.predict_proba(test[FEATURES])[:, 1]


proba_all = score_test()


def rate_chart(series_rate, series_n, title, subtitle, baseline, color=C_BLUE,
               height=2.6, pct_fmt=True):
    """Horizontal late-rate bars with a baseline marker."""
    fig, ax = plt.subplots(figsize=(7.2, height))
    y = np.arange(len(series_rate))[::-1]
    ax.barh(y, series_rate.values, color=color, height=.62, zorder=3)
    ax.axvline(baseline, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([str(i)[:28] for i in series_rate.index], fontsize=8.5)
    for yi, (r, n) in zip(y, zip(series_rate.values, series_n.values)):
        ax.text(r + .012, yi, f"{r:.0%}   ({n:,})", va="center",
                fontsize=8, color=INK_2)
    ax.set_xlim(0, min(1.25, max(series_rate.values) * 1.30))
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    if pct_fmt:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(title, loc="left", pad=22)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=8.5,
            color=MUTED, va="bottom")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("DataCo late-delivery risk")
st.sidebar.caption("IT3091 · Group 2026-AI-05")

threshold = st.sidebar.slider(
    "Decision threshold", 0.20, 0.80, 0.40, 0.01,
    help="An order is flagged when its predicted probability reaches this value.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**What this app is**\n\n"
    "Our recommendation is *not* to deploy this model. It is to fix the "
    "delivery promises.\n\n"
    "The model is the evidence for that. This app shows the evidence."
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Training rows: {len(train):,}  \n"
    f"Test rows: {len(test):,}  \n"
    f"Test period: {pd.to_datetime(test[DATE_COL]).min():%b %Y} - "
    f"{pd.to_datetime(test[DATE_COL]).max():%b %Y}  \n"
    f"Seed: {manifest['random_state']}"
)

st.title("Why DataCo's orders arrive late")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Orders analysed", f"{len(train) + len(test):,}")
k2.metric("Arrive late", f"{train[TARGET].mean():.1%}")
k3.metric("First Class late rate", "100.0%")
k4.metric("Best model (test ROC-AUC)", "0.7807")

tabs = st.tabs(["Findings", "Score an order", "Try your own",
                "Model performance", "Recommendation"])

# ==========================================================================
# 1 - FINDINGS
# ==========================================================================
with tabs[0]:
    st.subheader("The delivery promise causes the lateness, not the network")
    st.caption("All charts on this tab use the training set only "
               f"({len(train):,} shipped orders, Jan 2015 - Oct 2016), which is "
               "how the findings were established.")

    base = train[TARGET].mean()
    g = (train.groupby("Shipping Mode")[TARGET]
              .agg(["mean", "size"]).loc[MODE_ORDER])
    st.pyplot(rate_chart(g["mean"], g["size"],
                         "First Class is late 100% of the time",
                         f"Late rate by shipping mode · dashed line = overall {base:.0%}",
                         base, C_BLUE, 2.7), use_container_width=True)

    st.markdown("**Promised versus actual transit time**")
    st.dataframe(pd.DataFrame({
        "Shipping mode": MODE_ORDER,
        "Promised": ["1 day", "2 days", "0 days", "4 days"],
        "Actual (min-median-max)": ["2 - 2 - 2", "2 - 4 - 6", "0 - 0 - 1", "2 - 4 - 6"],
        "Late rate": ["100.0%", "79.9%", "45.9%", "39.9%"],
    }), hide_index=True, use_container_width=True)

    st.info(
        "**Second Class and Standard Class have identical actual transit times** "
        "(2 to 6 days, median 4), yet one is late 80% of the time and the other "
        "40%. The only difference is the promise printed on the tier. "
        "First Class promises one day and always takes two, so it cannot succeed."
    )

    st.markdown("---")
    st.subheader("Nothing else explains it")

    left, right = st.columns(2)
    with left:
        dim = st.selectbox("Break the late rate down by",
                           ["Market", "Order Region", "Customer Segment",
                            "Department Name", "Type"])
        gg = (train.groupby(dim)[TARGET].agg(["mean", "size"])
                   .sort_values("mean", ascending=False).head(12))
        st.pyplot(rate_chart(gg["mean"], gg["size"], f"Late rate by {dim}",
                             "every bar sits close to the overall rate",
                             base, C_AQUA, max(2.4, 0.26 * len(gg) + 1.3)),
                  use_container_width=True)

    with right:
        hr = train.groupby("order_hour")[TARGET].mean()
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        colors = [C_ORANGE if h >= 12 else C_BLUE for h in hr.index]
        ax.bar(hr.index, hr.values, color=colors, width=.72, zorder=3)
        ax.axhline(base, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.axvline(11.5, color=INK, linewidth=1.4)
        ax.text(11.8, 0.645, "midday cut-off", fontsize=8.5, color=INK)
        ax.set_ylim(0.45, 0.68)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("hour the order was placed")
        ax.set_title("Orders placed after noon are late more often", loc="left", pad=22)
        ax.text(0, 1.03, "54.2% before noon, 60.6% after", transform=ax.transAxes,
                fontsize=8.5, color=MUTED, va="bottom")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Does the mode effect survive filtering?")
    st.caption("Filter the data and watch the four bars keep their order and "
               "roughly their heights. That is why one global SLA change fixes "
               "it everywhere, rather than needing regional work.")

    f1, f2 = st.columns(2)
    mk = f1.multiselect("Markets", sorted(train["Market"].unique()),
                        default=sorted(train["Market"].unique()))
    sg = f2.multiselect("Customer segments", sorted(train["Customer Segment"].unique()),
                        default=sorted(train["Customer Segment"].unique()))

    sub = train[train["Market"].isin(mk) & train["Customer Segment"].isin(sg)]
    if len(sub) < 50:
        st.warning("Too few orders in that selection to be meaningful.")
    else:
        gs = (sub.groupby("Shipping Mode")[TARGET].agg(["mean", "size"])
                 .reindex([m for m in MODE_ORDER if m in sub["Shipping Mode"].unique()]))
        st.pyplot(rate_chart(gs["mean"], gs["size"],
                             f"Late rate by shipping mode · {len(sub):,} orders selected",
                             "the pattern is the same in every slice",
                             sub[TARGET].mean(), C_VIOLET, 2.7),
                  use_container_width=True)

# ==========================================================================
# 2 - SCORE AN ORDER
# ==========================================================================
with tabs[1]:
    st.subheader("Score a real order from the held-out test set")

    modes = st.multiselect("Filter by shipping mode",
                           sorted(test["Shipping Mode"].unique()),
                           default=sorted(test["Shipping Mode"].unique()),
                           key="t2modes")
    subset = test[test["Shipping Mode"].isin(modes)] if modes else test

    if subset.empty:
        st.warning("No orders match that filter.")
    else:
        n = min(len(subset), 300)
        i = st.slider("Order", 0, n - 1, 0, key="t2idx")
        row = subset.iloc[i]
        p = float(model.predict_proba(row[FEATURES].to_frame().T)[:, 1][0])
        flagged, actual = p >= threshold, bool(row[TARGET])

        a, b = st.columns(2)
        with a:
            st.markdown("**Order details**")
            st.dataframe(pd.DataFrame({
                "Field": ["Order Id", "Placed", "Shipping Mode", "Market",
                          "Region", "Category", "Segment", "Price",
                          "Quantity", "Order total"],
                "Value": [str(row["Order Id"]),
                          f"{pd.to_datetime(row[DATE_COL]):%d %b %Y, %H:%M}",
                          row["Shipping Mode"], row["Market"], row["Order Region"],
                          row["Category Name"], row["Customer Segment"],
                          f"${row['Product Price']:.2f}",
                          str(int(row["Order Item Quantity"])),
                          f"${row['Order Item Total']:.2f}"],
            }), hide_index=True, use_container_width=True)

        with b:
            st.markdown("**Prediction**")
            st.progress(min(p, 1.0))
            st.markdown(f"## {p:.1%} chance of being late")
            if flagged:
                st.warning(f"FLAGGED at the {threshold:.2f} threshold")
            else:
                st.success(f"Not flagged at the {threshold:.2f} threshold")

            st.markdown("**What actually happened**")
            if actual:
                st.error("Delivered late")
            else:
                st.info("Arrived on time")

            if flagged == actual:
                st.caption("Correct.")
            elif flagged and not actual:
                st.caption("False alarm. Cost: one unnecessary shipping upgrade.")
            else:
                st.caption("Missed. Cost: an unhappy customer.")

# ==========================================================================
# 3 - TRY YOUR OWN
# ==========================================================================
with tabs[2]:
    st.subheader("Build an order and score it")
    st.caption("Starts from a real test order. Change Shipping Mode first, it "
               "moves the prediction more than everything else combined.")

    row = test.iloc[0].copy()
    a, b, c = st.columns(3)
    with a:
        row["Shipping Mode"] = st.selectbox("Shipping Mode", MODE_ORDER, index=3)
        row["Market"] = st.selectbox("Market", sorted(test["Market"].unique()))
    with b:
        row["Customer Segment"] = st.selectbox(
            "Customer Segment", sorted(test["Customer Segment"].unique()))
        row["Category Name"] = st.selectbox(
            "Product category", sorted(test["Category Name"].unique()))
    with c:
        hour = st.slider("Hour placed", 0, 23, 14)
        row["order_hour"] = hour
        row["is_afternoon"] = int(hour >= 12)
        row["Product Price"] = st.number_input(
            "Product price ($)", 1.0, 2000.0, float(row["Product Price"]), 10.0)

    row["Order Item Total"] = (row["Product Price"] * row["Order Item Quantity"]
                               - row["Order Item Discount"])
    p = float(model.predict_proba(row[FEATURES].to_frame().T)[:, 1][0])

    st.markdown("---")
    m1, m2 = st.columns([1, 2])
    m1.metric("Probability of late delivery", f"{p:.1%}")
    m2.progress(min(p, 1.0))
    if p >= threshold:
        st.warning(f"FLAGGED at the {threshold:.2f} threshold")
    else:
        st.success(f"Not flagged at the {threshold:.2f} threshold")

    st.markdown("**The same order on every shipping mode**")
    comp = []
    for m in MODE_ORDER:
        r = row.copy(); r["Shipping Mode"] = m
        comp.append({"Shipping Mode": m,
                     "Predicted late": float(model.predict_proba(
                         r[FEATURES].to_frame().T)[:, 1][0])})
    cdf = pd.DataFrame(comp).set_index("Shipping Mode")
    st.dataframe(cdf.style.format({"Predicted late": "{:.1%}"}),
                 use_container_width=True)
    st.caption("Everything else about the order is identical. Only the "
               "promised service level changes.")

# ==========================================================================
# 4 - MODEL PERFORMANCE
# ==========================================================================
with tabs[3]:
    st.subheader(f"Performance on all {len(test):,} held-out test orders")

    y = test[TARGET].to_numpy()
    pred = (proba_all >= threshold).astype(int)

    m = st.columns(5)
    m[0].metric("ROC-AUC", f"{roc_auc_score(y, proba_all):.4f}",
                help="Threshold-free, so the slider does not change it.")
    m[1].metric("Recall", f"{recall_score(y, pred):.3f}")
    m[2].metric("Precision", f"{precision_score(y, pred, zero_division=0):.3f}")
    m[3].metric("F1", f"{f1_score(y, pred):.3f}")
    m[4].metric("Accuracy", f"{accuracy_score(y, pred):.3f}")

    late_total, caught = int(y.sum()), int(((proba_all >= threshold) & (y == 1)).sum())
    st.info(f"At a threshold of **{threshold:.2f}** the model flags "
            f"**{pred.mean():.0%}** of all orders and catches **{caught:,} of "
            f"{late_total:,}** late ones ({caught/late_total:.0%}), missing "
            f"{late_total - caught:,}.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Model comparison (validation)**")
        st.dataframe(pd.DataFrame({
            "Model": ["Random Forest (tuned)", "Random Forest", "XGBoost",
                      "Decision Tree", "Logistic Regression (baseline)",
                      "Neural network", "Shipping Mode rule", "Majority class"],
            "ROC-AUC": [0.7704, 0.7694, 0.7676, 0.7673, 0.7471, 0.7407,
                        0.7387, 0.5000],
        }), hide_index=True, use_container_width=True)
        st.caption("The top three are statistically tied: the gap between them "
                   "(0.0022) is smaller than the fold-to-fold variation (0.0030).")

    with right:
        st.markdown("**Confusion matrix at the current threshold**")
        cm = confusion_matrix(y, pred)
        st.dataframe(pd.DataFrame(cm,
                                  index=["Actually on time", "Actually late"],
                                  columns=["Predicted on time", "Predicted late"]),
                     use_container_width=True)

    st.markdown("**How the threshold changes the trade-off**")
    rows = []
    for t in np.arange(0.20, 0.81, 0.02):
        pr = (proba_all >= t).astype(int)
        rows.append({"threshold": round(float(t), 2),
                     "flagged": pr.mean(),
                     "recall": recall_score(y, pr),
                     "precision": precision_score(y, pr, zero_division=0)})
    st.line_chart(pd.DataFrame(rows).set_index("threshold"))
    st.caption("The cliff is real. Predicted probabilities form a few clusters "
               "because Shipping Mode dominates, so you choose between a handful "
               "of operating points rather than tuning a dial.")

# ==========================================================================
# 5 - RECOMMENDATION
# ==========================================================================
with tabs[4]:
    st.subheader("What we recommend DataCo does")

    st.markdown("""
#### 1. Re-baseline the expedited shipping tiers

First Class promises one day and delivers in exactly two, every single time, so it fails
100% of the time by definition rather than by accident.

Second Class and Standard Class have the same actual transit times, 2 to 6 days with a
median of 4, but one promises 2 days and the other 4. One is late 80% of the time and the
other 40%.

**Action:** change First Class to a 2-day promise and review Second Class against its real
4-day median. On the training data this alone moves First Class from 100% late to roughly
0% late, with no operational change at all.

#### 2. Publicise the midday dispatch cut-off

Orders placed from 12:00 onward are late 60.6% of the time against 54.2% before noon.

**Action:** show the cut-off at checkout, and consider a second afternoon dispatch for
expedited orders.

#### 3. Use the model for triage, not as the headline

**Action:** run it at a threshold of 0.40 rather than the default 0.50, catching 79% of late
orders instead of 60%.

The model adds about 0.03 ROC-AUC over a rule that uses only the shipping mode. Most of its
value is confirming where the problem is, not sorting individual orders.
    """)

    st.warning(
        "**Limitations.** The dataset has no carrier, warehouse, route, weather "
        "or capacity data, so the ceiling of about 0.78 ROC-AUC is a property of "
        "the data rather than the method. The target is a derived arithmetic flag, "
        "so it inherits any error in the recorded shipping dates. Store locations "
        "are Puerto Rico and the USA only. Day-of-week order volume is almost "
        "perfectly uniform, which suggests the timestamps may be synthetic, so the "
        "midday cut-off finding should be checked against real dispatch records "
        "before acting on it."
    )

st.markdown("---")
st.caption("Built from `01_Preprocessing_FINAL.ipynb` and `02_Model_Training.ipynb`. "
           "The test set was opened once, at the end of the modelling notebook.")
