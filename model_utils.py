"""Helper functions for the hotel-cancellation Streamlit app (no Streamlit code in here,
so the logic can be tested on its own)."""
import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hotel_cancellation_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

COUNTRY_NAMES = {
    "PRT": "Portugal", "GBR": "United Kingdom", "FRA": "France", "ESP": "Spain", "DEU": "Germany",
    "ITA": "Italy", "IRL": "Ireland", "BEL": "Belgium", "BRA": "Brazil", "NLD": "Netherlands",
    "USA": "United States", "CHE": "Switzerland", "CN": "China (code CN)", "CHN": "China",
    "AUT": "Austria", "SWE": "Sweden", "POL": "Poland", "RUS": "Russia", "NOR": "Norway",
    "ROU": "Romania", "FIN": "Finland", "ISR": "Israel", "DNK": "Denmark", "AUS": "Australia",
    "AGO": "Angola", "LUX": "Luxembourg", "MAR": "Morocco", "TUR": "Turkey", "ARG": "Argentina",
    "HUN": "Hungary", "JPN": "Japan", "IND": "India", "CZE": "Czechia", "Unknown": "Unknown",
}
OTHER_COUNTRY = "Other country"

MEAL_LABELS = {"BB": "Bed & breakfast (BB)", "HB": "Half board (HB)", "FB": "Full board (FB)",
               "SC": "No meal (SC)", "Undefined": "Undefined"}
DEPOSIT_LABELS = {"No Deposit": "No deposit", "Non Refund": "Non-refundable deposit",
                  "Refundable": "Refundable deposit"}

# Groups of model features that belong to one thing a user can change (used for the explanation)
GROUPS = {
    "Lead time": ["lead_time"],
    "Deposit type": ["deposit_type"],
    "Market segment / channel": ["market_segment", "distribution_channel"],
    "Travel agent": ["agent"],
    "Company booking": ["has_company"],
    "Guest country": ["country"],
    "Hotel": ["hotel"],
    "Customer type": ["customer_type"],
    "Price (ADR)": ["adr"],
    "Special requests": ["total_of_special_requests"],
    "Parking spaces": ["required_car_parking_spaces"],
    "Booking changes": ["booking_changes"],
    "Guest history": ["is_repeated_guest", "previous_cancellations",
                      "previous_bookings_not_canceled", "total_previous_bookings"],
    "Length of stay": ["stays_in_weekend_nights", "stays_in_week_nights", "total_nights"],
    "Party size": ["adults", "children", "babies", "total_guests"],
    "Meal plan": ["meal"],
    "Room type": ["reserved_room_type"],
    "Arrival date": ["arrival_date_month", "arrival_date_week_number", "arrival_date_day_of_month"],
    "Days on waiting list": ["days_in_waiting_list"],
}

RISK_HIGH = 0.60   # at or above this probability a booking is shown as "High" risk


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


def defaults():
    d = date.today()
    return dict(
        hotel="City Hotel", arrival_date=d + timedelta(days=45), lead_time=60,
        stays_weekend=1, stays_week=2, adults=2, children=0, babies=0, meal="BB",
        country="PRT", market_segment="Online TA", distribution_channel="TA/TO",
        customer_type="Transient", deposit_type="No Deposit", agent="9", has_company=False,
        reserved_room_type="A", adr=100.0, parking=0, special_requests=1, booking_changes=0,
        waiting_days=0, is_repeated_guest=False, prev_cancellations=0, prev_not_canceled=0)


def presets():
    """Example bookings for a quick demo. 'offset' = days from today until arrival."""
    return {
        "Early non-refundable online booking (high risk)": dict(
            hotel="City Hotel", offset=200, lead_time=200, stays_weekend=1, stays_week=3, adults=2,
            children=0, babies=0, meal="BB", country="PRT", market_segment="Online TA",
            distribution_channel="TA/TO", customer_type="Transient", deposit_type="Non Refund",
            agent="9", adr=120.0, parking=0, special_requests=0, booking_changes=0),
        "Last-minute direct booking with requests (low risk)": dict(
            hotel="Resort Hotel", offset=5, lead_time=5, stays_weekend=1, stays_week=2, adults=2,
            children=0, babies=0, meal="BB", country="GBR", market_segment="Direct",
            distribution_channel="Direct", customer_type="Transient", deposit_type="No Deposit",
            agent="0", adr=95.0, parking=1, special_requests=2, booking_changes=1),
        "Typical online booking, no special requests (medium risk)": dict(
            hotel="City Hotel", offset=60, lead_time=60, stays_weekend=2, stays_week=3, adults=2,
            children=0, babies=0, meal="BB", country="PRT", market_segment="Online TA",
            distribution_channel="TA/TO", customer_type="Transient", deposit_type="No Deposit",
            agent="9", adr=110.0, parking=0, special_requests=0, booking_changes=0),
    }


def country_label(code):
    return f"{COUNTRY_NAMES.get(code, code)} ({code})" if code in COUNTRY_NAMES and code != "Unknown" \
        else ("Unknown" if code == "Unknown" else code)


def validate_inputs(i):
    """Return (errors, warnings). Errors stop the prediction."""
    errors, warnings = [], []
    guests = i["adults"] + i["children"] + i["babies"]
    nights = i["stays_weekend"] + i["stays_week"]
    if guests < 1:
        errors.append("The booking needs at least one guest.")
    if nights < 1:
        errors.append("The stay must be at least one night (weekend + week nights).")
    if i["adults"] == 0 and (i["children"] + i["babies"]) > 0:
        warnings.append("Children or babies without an adult is unusual; check the party size.")
    if i["babies"] > i["adults"]:
        warnings.append("More babies than adults is unusual; check the party size.")
    if i["is_repeated_guest"] is False and (i["prev_cancellations"] + i["prev_not_canceled"]) > 0:
        warnings.append("Previous bookings are entered but the guest is not marked as a repeat guest.")
    if i["adr"] == 0:
        warnings.append("A price of 0 is rare in the training data (mostly complimentary stays).")
    if i["lead_time"] > 365:
        warnings.append("Lead times above one year are rare; the prediction is less reliable.")
    if i["deposit_type"] == "Non Refund" and i["market_segment"] == "Groups":
        warnings.append("Non-refundable group bookings are very rare in the training data.")
    return errors, warnings


def build_row(i, meta):
    """Turn the user inputs into one row with exactly the columns the model was trained on."""
    d = i["arrival_date"]
    iso_week = int(d.isocalendar()[1])
    row = dict(
        hotel=i["hotel"], lead_time=int(i["lead_time"]),
        arrival_date_month=MONTHS[d.month - 1], arrival_date_week_number=iso_week,
        arrival_date_day_of_month=d.day,
        stays_in_weekend_nights=int(i["stays_weekend"]), stays_in_week_nights=int(i["stays_week"]),
        adults=int(i["adults"]), children=int(i["children"]), babies=int(i["babies"]),
        meal=i["meal"], country=i["country"], market_segment=i["market_segment"],
        distribution_channel=i["distribution_channel"], is_repeated_guest=int(bool(i["is_repeated_guest"])),
        previous_cancellations=int(i["prev_cancellations"]),
        previous_bookings_not_canceled=int(i["prev_not_canceled"]),
        reserved_room_type=i["reserved_room_type"], booking_changes=int(i["booking_changes"]),
        deposit_type=i["deposit_type"], agent=str(i["agent"]),
        days_in_waiting_list=int(i["waiting_days"]), customer_type=i["customer_type"],
        adr=float(i["adr"]), required_car_parking_spaces=int(i["parking"]),
        total_of_special_requests=int(i["special_requests"]),
        total_guests=int(i["adults"] + i["children"] + i["babies"]),
        total_nights=int(i["stays_weekend"] + i["stays_week"]),
        total_previous_bookings=int(i["prev_cancellations"] + i["prev_not_canceled"]),
        has_company=int(bool(i["has_company"])),
    )
    if "arrival_date_year" in meta["feature_order"]:
        row["arrival_date_year"] = int(meta.get("app_year", 2017))   # Stage 6: the app uses 2017
    return pd.DataFrame([row])[meta["feature_order"]]


def risk_level(p, threshold=0.35):
    """Low: below the decision threshold. Medium: flagged, but below 60%. High: 60% or more."""
    if p < threshold:
        return "Low"
    if p < RISK_HIGH:
        return "Medium"
    return "High"


ACTIONS = {
    "Low": "No special action needed. Treat the room as likely to be occupied.",
    "Medium": "Send a confirmation or reminder before arrival and keep an eye on this booking.",
    "High": ("Contact the guest to confirm, consider requesting a deposit or offering a flexible "
             "change, and factor this booking into overbooking decisions."),
}


def explain(model, row, meta, n_perm=40, seed=0):
    """Shapley-style explanation of one prediction.

    Starting from a 'typical' reference booking (most common category / median number from the training
    data), the factors of the real booking are switched on one by one in random order and the change in
    predicted probability is credited to the factor just switched on. Averaging over many random orders
    gives each factor's contribution; the contributions add up exactly to
    (this booking's probability - the typical booking's probability).
    Positive = pushes cancellation risk up, negative = pushes it down."""
    import numpy as np
    rng = np.random.default_rng(seed)
    ref = meta["reference"]
    names = list(GROUPS)
    actual = row.iloc[0].to_dict()
    cur_template = {c: ref[c] for c in meta["feature_order"]}
    rows, index = [dict(cur_template)], []
    for _ in range(n_perm):
        cur = dict(cur_template)
        for g in rng.permutation(len(names)):
            for c in GROUPS[names[g]]:
                cur[c] = actual[c]
            rows.append(dict(cur))
            index.append(g)
    frame = pd.DataFrame(rows)[meta["feature_order"]]
    for c in meta["cat_cols"]:
        frame[c] = frame[c].astype(str)
    probs = model.predict_proba(frame)[:, 1]
    p_ref = float(probs[0])
    contrib = np.zeros(len(names))
    k = 1
    for _ in range(n_perm):
        prev = p_ref
        for _g in range(len(names)):
            g = index[k - 1]
            contrib[g] += probs[k] - prev
            prev = probs[k]
            k += 1
    contrib /= n_perm
    p0 = float(model.predict_proba(row)[:, 1][0])
    out = pd.DataFrame({"factor": names, "effect": contrib})
    out = out.reindex(out["effect"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return p0, p_ref, out


def business_value(p, adr, nights, resale_rate):
    stay_value = float(adr) * int(nights)
    expected_loss = p * stay_value
    return dict(stay_value=stay_value, revenue_at_risk=expected_loss,
                net_loss_after_resale=expected_loss * (1 - resale_rate))


def template_frame(meta):
    """A small CSV template for batch scoring (one row per example booking)."""
    rows = []
    for name, p in presets().items():
        inp = defaults()
        inp.update({k: v for k, v in p.items() if k != "offset"})
        inp["arrival_date"] = date.today() + timedelta(days=p["offset"])
        rows.append(build_row(inp, meta))
    return pd.concat(rows, ignore_index=True)


def score_batch(model, df, meta, threshold=0.35):
    """Validate an uploaded table and score it. Returns (result_df, problems)."""
    need = meta["feature_order"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        return None, [f"Missing required columns: {', '.join(missing)}"]
    data = df[need].copy()
    problems = []
    if data.isnull().any().any():
        bad = data.isnull().any(axis=1).sum()
        return None, [f"{bad} row(s) contain empty values. Please fill or remove them."]
    for c in meta["num_cols"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    if data[meta["num_cols"]].isnull().any().any():
        return None, ["Some numeric columns contain non-numeric values."]
    for c in meta["cat_cols"]:
        data[c] = data[c].astype(str)
    if (data["total_guests"] < 1).any() or (data["total_nights"] < 1).any():
        problems.append("Some rows have zero guests or zero nights; their scores are unreliable.")
    p = model.predict_proba(data)[:, 1]
    out = df.copy()
    out["cancellation_probability"] = p.round(4)
    out["risk_level"] = [risk_level(x, threshold) for x in p]
    out["flagged_as_likely_cancel"] = (p >= threshold).astype(int)
    return out, problems
