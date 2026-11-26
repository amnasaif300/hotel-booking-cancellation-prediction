"""Hotel Booking Cancellation Predictor - Streamlit app (CDS 4003, Stage 7)."""
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_utils as mu  # noqa: E402

st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return mu.load_artifacts()


try:
    model, meta = get_artifacts()
except Exception as exc:  # model file missing or made with an incompatible scikit-learn version
    st.error("The model could not be loaded. Re-create it with `python src/train_model.py` "
             "using the scikit-learn version listed in requirements.txt.")
    st.exception(exc)
    st.stop()

CATS = meta["categories"]
RANGES = meta["ranges"]
DEFAULT_THRESHOLD = float(meta["threshold"])

# ---------------------------------------------------------------- session state defaults
DEFAULTS = mu.defaults()
for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)
st.session_state.setdefault("preset", "(none)")
PRESETS = mu.presets()


def apply_preset():
    name = st.session_state["preset"]
    if name in PRESETS:
        values = dict(DEFAULTS)
        values.update({k: v for k, v in PRESETS[name].items() if k != "offset"})
        values["arrival_date"] = date.today() + timedelta(days=PRESETS[name]["offset"])
        for k, v in values.items():
            st.session_state[k] = v


def reset_inputs():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["preset"] = "(none)"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Quick start")
    st.selectbox("Load an example booking", ["(none)"] + list(PRESETS), key="preset", on_change=apply_preset)
    st.button("Reset all inputs", on_click=reset_inputs)

    st.header("Hotel decision settings")
    threshold = st.slider(
        "Flag a booking as 'likely to cancel' at or above", 0.10, 0.90, DEFAULT_THRESHOLD, 0.05,
        help=f"Stage 6 chose {DEFAULT_THRESHOLD:.2f} because it gave the best F1 score on cross-validated "
             "training data. A lower value catches more cancellations but raises false alarms.")
    resale = st.slider("Chance a cancelled room is re-sold", 0, 100, 60, 5, format="%d%%",
                       help="Used only for the revenue estimate. Replace it with your hotel's own figure.") / 100
    st.caption("Model: Gradient Boosting (HistGradientBoosting), tuned in Stage 6.")

st.title("Hotel Booking Cancellation Predictor")
st.write("Estimate how likely a booking is to be cancelled, see **why**, and decide what to do about it. "
         "Built for hotel revenue and reservation teams.")

tab_predict, tab_batch, tab_model, tab_about = st.tabs(
    ["Predict a booking", "Batch scoring", "Model performance", "About"])

# ================================================================ TAB 1: single booking
with tab_predict:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Booking details")

        st.markdown("**Stay**")
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Hotel", CATS["hotel"], key="hotel")
        c2.date_input("Arrival date", key="arrival_date", min_value=date.today(),
                      max_value=date.today() + timedelta(days=730),
                      help="Only month, week and day are used. The model uses 2017 as the arrival year "
                           "(the latest year in the training data).")
        c3.number_input("Lead time (days between booking and arrival)", 0, int(RANGES["lead_time"][1]),
                        step=1, key="lead_time")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Weekend nights", 0, 20, step=1, key="stays_weekend")
        c2.number_input("Week nights", 0, 50, step=1, key="stays_week")
        c3.selectbox("Meal plan", CATS["meal"], key="meal",
                     format_func=lambda x: mu.MEAL_LABELS.get(x, x))

        st.markdown("**Guests**")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Adults", 0, 10, step=1, key="adults")
        c2.number_input("Children", 0, 10, step=1, key="children")
        c3.number_input("Babies", 0, 10, step=1, key="babies")
        country_options = CATS["country"] + [mu.OTHER_COUNTRY]
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Guest country", country_options, key="country",
                     format_func=lambda x: x if x == mu.OTHER_COUNTRY else mu.country_label(x))
        c2.selectbox("Customer type", CATS["customer_type"], key="customer_type")
        c3.selectbox("Reserved room type", CATS["reserved_room_type"], key="reserved_room_type")

        st.markdown("**Booking channel and payment**")
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Market segment", CATS["market_segment"], key="market_segment")
        c2.selectbox("Distribution channel", CATS["distribution_channel"], key="distribution_channel")
        c3.selectbox("Deposit type", CATS["deposit_type"], key="deposit_type",
                     format_func=lambda x: mu.DEPOSIT_LABELS.get(x, x))
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Travel agent ID", CATS["agent"], key="agent",
                     format_func=lambda x: "No agent" if x == "0" else f"Agent {x}",
                     help="Agent IDs come from the dataset. 'No agent' means the booking was not made through an agent.")
        c2.number_input("Average daily rate (price per night)", 0.0, 1000.0, step=5.0, key="adr")
        c3.checkbox("Booked by a company", key="has_company")

        with st.expander("Requests and guest history"):
            c1, c2, c3 = st.columns(3)
            c1.number_input("Special requests", 0, 5, step=1, key="special_requests")
            c2.number_input("Parking spaces requested", 0, 8, step=1, key="parking")
            c3.number_input("Changes made to the booking", 0, 18, step=1, key="booking_changes",
                            help="Changes made before arrival. Use 0 for a brand-new booking.")
            c1, c2, c3 = st.columns(3)
            c1.checkbox("Repeat guest", key="is_repeated_guest")
            c2.number_input("Previous cancellations", 0, 26, step=1, key="prev_cancellations")
            c3.number_input("Previous stays not cancelled", 0, 72, step=1, key="prev_not_canceled")
            st.number_input("Days on the waiting list", 0, 400, step=1, key="waiting_days")

    inputs = {k: st.session_state[k] for k in DEFAULTS}
    errors, warnings = mu.validate_inputs(inputs)

    with right:
        st.subheader("Prediction")
        for e in errors:
            st.error(e)
        for w in warnings:
            st.warning(w)

        if not errors:
            row = mu.build_row(inputs, meta)
            p, p_ref, effects = mu.explain(model, row, meta)
            level = mu.risk_level(p, threshold)
            flagged = p >= threshold
            nights = inputs["stays_weekend"] + inputs["stays_week"]

            m1, m2 = st.columns(2)
            m1.metric("Cancellation probability", f"{p:.0%}" if p >= 0.01 else "<1%")
            m2.metric("Risk level", level)
            st.progress(min(max(p, 0.0), 1.0))
            if flagged:
                st.error(f"Flagged as **likely to cancel** (probability is at or above {threshold:.0%}).")
            else:
                st.success(f"Predicted to **stay** (probability is below {threshold:.0%}).")
            st.info("**Suggested action:** " + mu.ACTIONS[level])

            bv = mu.business_value(p, inputs["adr"], nights, resale)
            st.markdown("**Estimated revenue impact** (illustrative)")
            b1, b2, b3 = st.columns(3)
            b1.metric("Room revenue", f"{bv['stay_value']:,.0f}")
            b2.metric("Expected at risk", f"{bv['revenue_at_risk']:,.0f}")
            b3.metric("After re-sale", f"{bv['net_loss_after_resale']:,.0f}")
            st.caption("Room revenue = price per night x nights. Expected at risk = probability x room revenue. "
                       "After re-sale subtracts the share of cancelled rooms the hotel expects to sell again. "
                       "Same currency as the price you entered.")

            st.markdown("**Why this prediction?**")
            top = effects.head(7).iloc[::-1]
            fig, ax = plt.subplots(figsize=(5.6, 3.4))
            colors = ["#c0392b" if v > 0 else "#2e86c1" for v in top["effect"]]
            ax.barh(top["factor"], top["effect"] * 100, color=colors)
            ax.axvline(0, color="#555", linewidth=0.8)
            ax.set_xlabel("Change in cancellation probability (percentage points)")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                f"A typical booking in the training data has a {p_ref:.0%} cancellation probability. "
                "Red bars push this booking's risk above that, blue bars push it below. "
                "The bars add up to the difference between the two (Shapley-style attribution). "
                "This shows what the model relies on; it does not prove cause and effect.")
        else:
            st.info("Fix the input problems above to see a prediction.")

# ================================================================ TAB 2: batch scoring
with tab_batch:
    st.subheader("Score many bookings at once")
    st.write("Upload a CSV with one row per booking. The file must contain exactly the columns of the template "
             "(download it below, replace the example rows with your own bookings).")
    template = mu.template_frame(meta)
    st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"),
                       file_name="booking_template.csv", mime="text/csv")
    with st.expander("Columns the file needs"):
        st.write(", ".join(meta["feature_order"]))
    up = st.file_uploader("Upload bookings (CSV)", type="csv")
    if up is not None:
        try:
            data = pd.read_csv(up)
        except Exception as exc:
            st.error(f"The file could not be read as CSV: {exc}")
            data = None
        if data is not None:
            if len(data) > 20000:
                st.error("Please upload at most 20,000 rows at a time.")
            else:
                result, problems = mu.score_batch(model, data, meta, threshold)
                if result is None:
                    for pr in problems:
                        st.error(pr)
                else:
                    for pr in problems:
                        st.warning(pr)
                    counts = result["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Bookings scored", len(result))
                    k2.metric("Flagged likely to cancel", int(result["flagged_as_likely_cancel"].sum()))
                    k3.metric("High risk", int(counts["High"]))
                    k4.metric("Average probability", f"{result['cancellation_probability'].mean():.0%}")
                    shown = result.sort_values("cancellation_probability", ascending=False)
                    st.dataframe(shown, height=360)
                    st.download_button("Download scored bookings", shown.to_csv(index=False).encode("utf-8"),
                                       file_name="scored_bookings.csv", mime="text/csv")

# ================================================================ TAB 3: model performance
with tab_model:
    mt, md = meta["metrics"], meta["metrics_default_threshold"]
    st.subheader("How well does the model work?")
    st.write(f"Gradient Boosting, tuned with RandomizedSearchCV (5-fold CV) in Stage 6. Evaluated once on "
             f"{meta['n_test']:,} bookings the model never saw during training.")
    k = st.columns(5)
    k[0].metric("ROC-AUC", f"{mt['roc_auc']:.3f}")
    k[1].metric(f"Recall @ {meta['threshold']:.2f}", f"{mt['recall']:.1%}")
    k[2].metric(f"Precision @ {meta['threshold']:.2f}", f"{mt['precision']:.1%}")
    k[3].metric(f"F1 @ {meta['threshold']:.2f}", f"{mt['f1']:.3f}")
    k[4].metric(f"Accuracy @ {meta['threshold']:.2f}", f"{mt['accuracy']:.1%}")

    st.markdown("**Default threshold (0.50) vs the chosen threshold**")
    cmp_df = pd.DataFrame({
        "Threshold": [0.50, meta["threshold"]],
        "Precision": [md["precision"], mt["precision"]], "Recall": [md["recall"], mt["recall"]],
        "F1": [md["f1"], mt["f1"]], "Accuracy": [md["accuracy"], mt["accuracy"]],
        "False alarms": [md["fp"], mt["fp"]], "Missed cancellations": [md["fn"], mt["fn"]]})
    st.dataframe(cmp_df.style.format({"Threshold": "{:.2f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
                                      "F1": "{:.3f}", "Accuracy": "{:.3f}"}), hide_index=True)
    st.caption("Lowering the threshold catches more real cancellations (fewer missed) at the cost of more false alarms.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Most important features** (permutation importance)")
        imp = pd.DataFrame(meta["importance"]).head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        ax.barh(imp["feature"], imp["importance"], color="#2e86c1")
        ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        st.markdown("**Are the probabilities trustworthy?** (calibration)")
        cal = pd.DataFrame(meta["calibration"])
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        ax.plot([0, 1], [0, 1], "--", color="#999", label="Perfect")
        ax.plot(cal["mean_predicted"], cal["actual_rate"], "o-", color="#c0392b", label="Model")
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Share that actually cancelled")
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Points close to the dashed line mean a predicted 70% really does cancel about 70% of the time.")

    st.markdown("**Model comparison from Stages 5 and 6** (test set)")
    comp = pd.DataFrame([
        ("Baseline (majority class)", "Stage 5", 0.000, 0.500),
        ("Logistic Regression", "Stage 5", 0.5826, 0.8561),
        ("Decision Tree", "Stage 5", 0.6504, 0.8907),
        ("Random Forest", "Stage 5", 0.7012, 0.9091),
        ("Gradient Boosting", "Stage 5", 0.7018, 0.9143),
        ("Random Forest (tuned)", "Stage 6", 0.7019, 0.9120),
        ("Gradient Boosting (tuned) - deployed", "Stage 6", 0.7041, 0.9144),
    ], columns=["Model", "Stage", "F1 (threshold 0.5)", "ROC-AUC"])
    st.dataframe(comp, hide_index=True)

    st.markdown("**Limitations**")
    st.markdown(
        "- Trained on bookings from two hotels in Portugal (2015-2017). Behaviour and policies may have changed, "
        "and other hotels may differ.\n"
        f"- Duplicate rows were removed for modelling, so the training cancellation rate is "
        f"{meta['base_rate_train']:.1%} (full data: {meta['base_rate_full_data']:.1%}). Real-world precision and recall may differ.\n"
        "- The model uses 2017 as the arrival year because it was trained on 2015-2017 data.\n"
        "- Country and travel agent are strong predictors, but they may reflect these hotels' own channels and policies.\n"
        "- Features such as parking spaces and booking changes may change after the booking is made; "
        "use the latest values available at prediction time.\n"
        "- Predictions support staff decisions. They should not trigger automatic overbooking or refusals.")

# ================================================================ TAB 4: about
with tab_about:
    st.subheader("About this app")
    st.markdown(
        "**Business problem.** Cancelled bookings leave rooms empty and cost hotels revenue. If a hotel can "
        "tell which bookings are likely to be cancelled, it can send reminders, ask for deposits, plan staffing "
        "and manage overbooking more carefully.\n\n"
        "**What the app does.**\n"
        "1. *Predict a booking*: enter the details of one booking, get a cancellation probability, a risk level, "
        "a suggested action, an estimate of the revenue at risk and an explanation of the main drivers.\n"
        "2. *Batch scoring*: score a whole file of bookings and download the ranked list.\n"
        "3. *Model performance*: see how well the model works and where it falls short.\n\n"
        "**How it works.** A Gradient Boosting model (many small decision trees built one after another, each "
        "correcting the previous ones) was trained on 85,975 historical bookings. It reads booking features known "
        "around the time of reservation (lead time, deposit type, market segment, country, price, history and so on) "
        "and outputs the probability of cancellation.\n\n"
        "**Data.** Hotel booking demand dataset (City Hotel and Resort Hotel, Portugal), described in Antonio, "
        "de Almeida and Nunes (2019), *Hotel booking demand datasets*, Data in Brief 22, 41-49. "
        "See the project repository for the source and licence.\n\n"
        "**Project.** CDS 4003 Data Science for Business Application, Stage 7 (deployment). "
        "Code, notebooks and documentation are in the project's GitHub repository.")
