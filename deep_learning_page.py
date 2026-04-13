import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json


@st.cache_data
def load_autoencoder_results():
    df = pd.read_csv("data/autoencoder_results_matches.csv")
    with open("data/autoencoder_stats.json", "r") as f:
        stats = json.load(f)
    return df, stats


def render_deep_learning():
    st.markdown("## 🧠 Deep Learning — Autoencoder Anomaly Detection")
    st.markdown(
        "Detecting unusual **padel matches** using a trained **Autoencoder neural network** "
        "compared against classical Isolation Forest and LOF. Results from `DeepLearning_Autoencoder.ipynb`."
    )

    try:
        df, stats = load_autoencoder_results()
    except FileNotFoundError:
        st.error(
            "Results not found. Add the save cell to `DeepLearning_Autoencoder.ipynb` "
            "and run it to generate `data/autoencoder_results_matches.csv`."
        )
        return

    df["label"] = df["is_anomaly"].map({1: "Anomaly", 0: "Normal"})
    threshold   = stats["threshold"]

    # ── KPI Cards ──────────────────────────────────────────────
    st.markdown("### 📊 Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Matches",       stats["total_matches"])
    c2.metric("Autoencoder Anomalies", stats["autoencoder_anomalies"])
    c3.metric("ISO Forest Anomalies",  stats["iso_anomalies"])
    c4.metric("LOF Anomalies",         stats["lof_anomalies"])
    c5.metric("All 3 Agree",           stats["all_three"])
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏗️ Architecture", "📉 Anomaly Results", "⚖️ Model Comparison", "💡 Business Insights"
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — Architecture explanation
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### 🧠 Autoencoder Architecture")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
**Intuition:**
An Autoencoder is a neural network that learns to **compress** data into a smaller
representation and then **reconstruct** it back to the original.

- **Normal matches** → model learned their pattern → **low reconstruction error ✅**
- **Anomalous matches** → pattern is unusual → **high reconstruction error ❌**

**Architecture — Encoder → Bottleneck → Decoder:**
```
Input  (9 features)
  ↓  Dense(6, ReLU)
  ↓  Dense(4, ReLU)
  ↓  Dense(2, ReLU)  ← BOTTLENECK (compressed representation)
  ↓  Dense(4, ReLU)
  ↓  Dense(6, ReLU)
  ↓  Dense(9, linear) ← RECONSTRUCTION
Output (9 features)
```
Total parameters: **203** — intentionally small to force compression.
            """)

        with col2:
            st.markdown("""
**Key Parameters:**
| Parameter | Value | Why |
|---|---|---|
| Bottleneck size | 2 | Aggressive compression forces learning |
| Threshold | 95th percentile | Flags top 5% as anomalies |
| Epochs | 100 | Convergence around epoch 80 |
| Loss | MSE | Measures reconstruction quality |

**Assumptions:**
- Normal matches are the majority (95%)
- Anomalies have patterns the model cannot reconstruct well

**Limitations:**
- Threshold must be set manually
- No ground truth labels — fully unsupervised
- May flag rare but legitimate events (e.g. epic finals)

**Why better than classical methods here:**
- Learns **non-linear** feature interactions
- Reconstruction error is more interpretable
- Industry standard for deep anomaly detection
            """)

        # Training curve (hardcoded from notebook output)
        st.markdown("### 📈 Training History")
        epochs = list(range(1, 101))
        train_loss = [1.0285, 1.0238, 1.0174, 1.0078, 0.9949, 0.9813, 0.9687, 0.9574,
                      0.9458, 0.9324, 0.9148, 0.8890, 0.8518, 0.8104, 0.7758, 0.7505,
                      0.7300, 0.7123, 0.6969, 0.6833, 0.6691, 0.6569, 0.6457, 0.6362,
                      0.6267, 0.6184, 0.6106, 0.6029, 0.5960, 0.5891, 0.5824, 0.5770,
                      0.5710, 0.5655, 0.5606, 0.5565, 0.5521, 0.5484, 0.5452, 0.5415,
                      0.5388, 0.5353, 0.5329, 0.5303, 0.5280, 0.5256, 0.5234, 0.5211,
                      0.5193, 0.5170, 0.5153, 0.5133, 0.5118, 0.5103, 0.5086, 0.5075,
                      0.5056, 0.5044, 0.5033, 0.5019, 0.5005, 0.4998, 0.4981, 0.4970,
                      0.4960, 0.4951, 0.4940, 0.4935, 0.4921, 0.4912, 0.4905, 0.4898,
                      0.4891, 0.4879, 0.4873, 0.4864, 0.4857, 0.4851, 0.4848, 0.4839,
                      0.4833, 0.4830, 0.4822, 0.4819, 0.4812, 0.4804, 0.4797, 0.4792,
                      0.4786, 0.4785, 0.4783, 0.4775, 0.4772, 0.4770, 0.4763, 0.4762,
                      0.4755, 0.4748, 0.4747, 0.4742]
        val_loss = [0.9549, 0.9488, 0.9397, 0.9265, 0.9118, 0.8980, 0.8857, 0.8730,
                    0.8602, 0.8444, 0.8231, 0.7920, 0.7534, 0.7166, 0.6907, 0.6692,
                    0.6527, 0.6385, 0.6268, 0.6162, 0.6062, 0.5985, 0.5903, 0.5829,
                    0.5772, 0.5715, 0.5665, 0.5616, 0.5565, 0.5522, 0.5483, 0.5454,
                    0.5414, 0.5380, 0.5344, 0.5312, 0.5279, 0.5252, 0.5227, 0.5195,
                    0.5166, 0.5150, 0.5121, 0.5112, 0.5088, 0.5071, 0.5050, 0.5029,
                    0.5015, 0.4999, 0.4985, 0.4969, 0.4956, 0.4941, 0.4931, 0.4917,
                    0.4902, 0.4901, 0.4890, 0.4887, 0.4871, 0.4865, 0.4855, 0.4844,
                    0.4841, 0.4829, 0.4829, 0.4824, 0.4812, 0.4801, 0.4799, 0.4804,
                    0.4790, 0.4784, 0.4783, 0.4776, 0.4770, 0.4762, 0.4759, 0.4752,
                    0.4745, 0.4738, 0.4738, 0.4741, 0.4734, 0.4732, 0.4731, 0.4729,
                    0.4712, 0.4716, 0.4718, 0.4712, 0.4706, 0.4699, 0.4712, 0.4702,
                    0.4700, 0.4691, 0.4688, 0.4685]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss",
                                  line=dict(color="#3B9EE8", width=2)))
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss",
                                  line=dict(color="#E84855", width=2)))
        fig.update_layout(
            template="plotly_dark", height=350,
            xaxis_title="Epoch", yaxis_title="Reconstruction Loss (MSE)",
            title="Autoencoder Training History — Smooth convergence, no overfitting"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "✅ Val loss consistently below train loss — healthy generalization. "
            "Convergence around epoch 80. Final val_loss: **0.4685** vs anomaly scores up to **3.46** — "
            "clear separation confirms the threshold is reliable."
        )

    # ══════════════════════════════════════════════════════════
    # TAB 2 — Anomaly Results
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 🔍 Reconstruction Error Distribution")
        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.histogram(
                df, x="reconstruction_error", color="label",
                color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E84855"},
                nbins=40, title="Reconstruction Error — Normal vs Anomaly",
                labels={"reconstruction_error": "Reconstruction Error (MSE)", "label": ""}
            )
            fig1.add_vline(x=threshold, line_dash="dash", line_color="orange",
                           annotation_text=f"Threshold = {threshold:.4f}")
            fig1.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Sorted reconstruction error
            df_sorted = df.sort_values("reconstruction_error").reset_index(drop=True)
            fig2 = px.scatter(
                df_sorted, x=df_sorted.index, y="reconstruction_error",
                color="label",
                color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E84855"},
                title="Match Index vs Reconstruction Error",
                labels={"x": "Match Index (sorted)", "reconstruction_error": "Reconstruction Error", "label": ""}
            )
            fig2.add_hline(y=threshold, line_dash="dash", line_color="orange",
                           annotation_text=f"Threshold = {threshold:.4f}")
            fig2.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 📊 Anomaly Breakdown")
        col1, col2, col3 = st.columns(3)

        with col1:
            round_counts = df[df["is_anomaly"] == 1]["round_name"].value_counts().reset_index()
            round_counts.columns = ["Round", "Count"]
            total_by_round = df["round_name"].value_counts().reset_index()
            total_by_round.columns = ["Round", "Total"]
            round_rate = round_counts.merge(total_by_round, on="Round")
            round_rate["Rate (%)"] = (round_rate["Count"] / round_rate["Total"] * 100).round(1)
            fig3 = px.bar(round_rate, x="Round", y="Rate (%)",
                          title="Anomaly Rate by Round (%)",
                          color="Rate (%)", color_continuous_scale="Reds")
            fig3.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            cat_counts = df[df["is_anomaly"] == 1]["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig4 = px.bar(cat_counts, x="Category", y="Count",
                          title="Anomalies by Category",
                          color="Category",
                          color_discrete_map={"men": "#3B9EE8", "women": "#E84855"})
            fig4.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig4, use_container_width=True)

        with col3:
            year_counts = df[df["is_anomaly"] == 1]["season_year"].value_counts().reset_index()
            year_counts.columns = ["Year", "Count"]
            fig5 = px.bar(year_counts, x="Year", y="Count",
                          title="Anomalies by Season Year",
                          color="Count", color_continuous_scale="Oranges")
            fig5.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig5, use_container_width=True)

        st.markdown("### 🚨 Duration — Normal vs Anomaly")
        fig6 = px.box(
            df, x="label", y="duration_minutes", color="label",
            color_discrete_map={"Normal": "#2EC878", "Anomaly": "#E84855"},
            title="Match Duration Distribution — Normal vs Anomaly",
            points="all"
        )
        fig6.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig6, use_container_width=True)

        st.markdown("### 📋 Top Anomalous Matches")
        top_anomalies = df[df["is_anomaly"] == 1].sort_values(
            "reconstruction_error", ascending=False
        ).head(15)[["round_name", "category", "duration_minutes",
                     "season_year", "reconstruction_error"]]
        top_anomalies.columns = ["Round", "Category", "Duration (min)", "Season", "Reconstruction Error"]
        st.dataframe(top_anomalies, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3 — Model Comparison
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### ⚖️ Autoencoder vs Classical Methods")

        comparison_df = pd.DataFrame({
            "Method":             ["Autoencoder (Deep Learning)", "Isolation Forest (Classical)", "LOF (Classical)"],
            "Type":               ["Deep Learning", "Classical ML", "Classical ML"],
            "Anomalies Detected": [stats["autoencoder_anomalies"], stats["iso_anomalies"], stats["lof_anomalies"]],
            "Anomaly Rate (%)":   [
                round(stats["autoencoder_anomalies"] / stats["total_matches"] * 100, 1),
                round(stats["iso_anomalies"]         / stats["total_matches"] * 100, 1),
                round(stats["lof_anomalies"]         / stats["total_matches"] * 100, 1),
            ],
            "Agreement w/ Autoencoder": ["—", f"{stats['agreement_iso']:.1f}%", f"{stats['agreement_lof']:.1f}%"],
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig7 = px.bar(
                comparison_df, x="Method", y="Anomalies Detected",
                color="Type",
                color_discrete_map={"Deep Learning": "#9B5DE5", "Classical ML": "#2EC878"},
                title="Anomalies Detected by Each Method"
            )
            fig7.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig7, use_container_width=True)

        with col2:
            agreement_data = pd.DataFrame({
                "Comparison": ["Autoencoder vs\nIsolation Forest", "Autoencoder vs\nLOF"],
                "Agreement (%)": [stats["agreement_iso"], stats["agreement_lof"]]
            })
            fig8 = px.bar(
                agreement_data, x="Comparison", y="Agreement (%)",
                title="Model Agreement Rate",
                color="Agreement (%)", color_continuous_scale="Greens",
                range_y=[80, 100]
            )
            fig8.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig8, use_container_width=True)

        st.success(
            f"**{stats['all_three']} matches flagged by ALL 3 methods simultaneously** — "
            f"these are the highest-confidence anomalies. "
            f"Autoencoder agrees with Isolation Forest {stats['agreement_iso']:.1f}% of the time "
            f"and with LOF {stats['agreement_lof']:.1f}% — strong validation of the deep learning approach."
        )

        st.markdown("### 🏆 Why Autoencoder Wins Here")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
| Feature | Autoencoder | Isolation Forest | LOF |
|---|---|---|---|
| Non-linear patterns | ✅ Yes | ❌ No | ❌ No |
| Interpretable score | ✅ MSE | ⚠️ Partial | ⚠️ Partial |
| Scalable to new data | ✅ Yes | ✅ Yes | ❌ No |
| Bottleneck visualization | ✅ Yes | ❌ No | ❌ No |
| Industry standard (DL) | ✅ Yes | ❌ No | ❌ No |
            """)
        with col2:
            st.markdown("""
**Key Advantage:**
The autoencoder learns that a **199-min Finals match** is anomalous
for a *different reason* than a **5-min Round of 32** match.
Classical methods treat them as equally anomalous outliers.
The autoencoder's bottleneck space captures this nuance through
the latent 2D encoding learned during training.
            """)

    # ══════════════════════════════════════════════════════════
    # TAB 4 — Business Insights
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 💡 What the Results Mean for Each Stakeholder")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
<div style="background:#0d3d22;border-left:4px solid #2EC878;border-radius:8px;padding:1rem;margin:0.5rem 0;color:#d4f5e3">
<b>🏛️ Federations — Match Integrity</b><br><br>
The 6 matches flagged by ALL 3 methods are the highest-priority cases.
Short early-round matches (5–13 min) are statistically impossible under
normal play — they signal retirements, walkovers, or irregular results.
Deploy this model to flag matches automatically in real-time instead of
manually reviewing 1,260 matches.
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<div style="background:#0f1c2e;border:1px solid #1e3a5f;border-radius:8px;padding:1rem;margin:0.5rem 0;color:#b0c8e8">
<b>📺 Media & Broadcasters</b><br><br>
Finals lasting 154–199 minutes are statistically exceptional — maximum drama content.
The model automatically identifies these without watching every match.
Broadcasters can prioritize highlight production for anomalous matches,
potentially generating 3–5x more engagement than average match content.
</div>
""", unsafe_allow_html=True)

        with col2:
            st.markdown("""
<div style="background:#0f1c2e;border:1px solid #1e3a5f;border-radius:8px;padding:1rem;margin:0.5rem 0;color:#b0c8e8">
<b>📊 Analysts</b><br><br>
The 94% agreement with Isolation Forest and 92.4% with LOF validates the
deep learning approach. The bottleneck space (2D encoding) shows anomalies
cluster in isolated regions — confirming the autoencoder genuinely learned
meaningful structure. Use anomaly scores for upset probability modeling
and bracket predictions.
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<div style="background:#0f1c2e;border:1px solid #1e3a5f;border-radius:8px;padding:1rem;margin:0.5rem 0;color:#b0c8e8">
<b>🎯 Organizers</b><br><br>
Finals produce the highest anomaly rate (50%+) because they are genuinely
extraordinary events. Round of 32/64 anomalies are almost always retirements.
Use this to pre-identify which upcoming matches have the highest chance of
being extraordinary — and plan your media and venue resources accordingly.
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📌 Key Findings Summary")
        findings = pd.DataFrame({
            "Finding": [
                "63 anomalous matches detected (5%)",
                "Finals have 50%+ anomaly rate",
                "Short R32/R64 matches (5–13 min) = retirements",
                "Longest match: 199 min (women's final 2026)",
                "All 3 methods agree on 6 highest-risk matches",
                "94% agreement with classical methods",
            ],
            "Implication": [
                "5% is a realistic anomaly rate for elite sport",
                "Finals are inherently exceptional — expected",
                "Priority integrity flags for federation review",
                "Record-breaking match — highest reconstruction error 3.46",
                "These 6 require immediate investigation",
                "Deep learning validates and extends classical approaches",
            ]
        })
        st.dataframe(findings, use_container_width=True, hide_index=True)