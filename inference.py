import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

# load the trained model and preprocessor
rf_model = joblib.load(r'notebook\rf_model.joblib')
preprocessor = joblib.load(r'notebook\preprocessor.joblib')

# initialize the FastAPI app
app = FastAPI(title="Fraudulent Transaction Detection API")

# define the input data model
class Transaction(BaseModel):
    # --- raw fields ---
    
    home_country: str = Field(..., json_schema_extra={"example": "us"})
    source_currency: str = Field(..., json_schema_extra={"example": "usd"})
    dest_currency: str = Field(..., json_schema_extra={"example": "cad"})
    channel: str = Field(..., json_schema_extra={"example": "mobile"})
    amount_src: float = Field(..., json_schema_extra={"example": 100.0})
    amount_usd: float = Field(..., json_schema_extra={"example": 100.0})
    fee: float = Field(..., json_schema_extra={"example": 4.25})
    exchange_rate_src_to_dest: float = Field(..., json_schema_extra={"example": 1.85})
    new_device: bool = Field(..., json_schema_extra={"example": False})
    ip_country: str = Field(..., json_schema_extra={"example": "us"})
    location_mismatch: bool = Field(..., json_schema_extra={"example": False})
    kyc_tier: str = Field(..., json_schema_extra={"example": "low"})

    # --- hard / provided-from-history fields ---
    account_age_days: int = Field(..., json_schema_extra={"example": 13})
    ip_risk_score: float = Field(..., json_schema_extra={"example": 0.1})
    device_trust_score: float = Field(..., json_schema_extra={"example": 0.4})
    chargeback_history_count: int = Field(..., json_schema_extra={"example": 0})
    risk_score_internal: float = Field(..., json_schema_extra={"example": 0.2})
    txn_velocity_1h: int = Field(..., json_schema_extra={"example": 1})
    txn_velocity_24h: int = Field(..., json_schema_extra={"example": 3})
    corridor_risk: float = Field(..., json_schema_extra={"example": 0.15})

        # --- engineered fields (caller must supply these too) ---
    hour_of_day: int = Field(..., json_schema_extra={"example": 14})
    day_of_week: int = Field(..., json_schema_extra={"example": 1})
    is_weekend: int = Field(..., json_schema_extra={"example": 0})
    is_night: bool = Field(..., json_schema_extra={"example": False})
    corridor: str = Field(..., json_schema_extra={"example": "us_eur"})
    age_bucket: str = Field(..., json_schema_extra={"example": "181-365d"})
    amount_bucket: str = Field(..., json_schema_extra={"example": "$100-500"})
    ip_risk_bucket: str = Field(..., json_schema_extra={"example": "Low (0-0.3)"})
    device_trust_bucket: str = Field(..., json_schema_extra={"example": "High>0.8"})
    night_hours: int = Field(..., json_schema_extra={"example": 0})
    account_very_new: int = Field(..., json_schema_extra={"example": 0})
    account_new: int = Field(..., json_schema_extra={"example": 0})
    velocity_burst: int = Field(..., json_schema_extra={"example": 0})
    amount_high: int = Field(..., json_schema_extra={"example": 0})
    ip_risk_high: int = Field(..., json_schema_extra={"example": 0})
    device_low: int = Field(..., json_schema_extra={"example": 0})

# This decorator registers a web address. Think of it as putting up a sign 
# that says "send transactions here." The "post" means people SEND us data 
# (a transaction to check), rather than just asking to read something back.
@app.post("/predict")

# This is the function that runs whenever someone hits that address.
# "transaction: Transaction" is us telling FastAPI: "I expect a transaction, 
# and it must match the Transaction blueprint I defined earlier." 
# If the data is malformed, FastAPI rejects it before this code even runs.
def predict_fraud(transaction: Transaction):

    # Our model was trained on a table (rows and columns), not on a single 
    # transaction object. So here we convert the incoming transaction into a 
    # one-row table. model_dump() turns it into a dictionary; wrapping it in 
    # [ ] and pd.DataFrame turns that into a table with exactly one row.
    df = pd.DataFrame([transaction.model_dump()])

    # During training, we cleaned and scaled the data (encoded categories, 
    # scaled numbers, etc.). The model only understands data in THAT form. 
    # So we run the new transaction through the same preprocessor. We use 
    # .transform (not .fit_transform) because we are NOT re-learning anything 
    # from one transaction — we just apply what was already learned.
    df_processed = preprocessor.transform(df)

    # Now the actual prediction. The model returns a list of answers (one per 
    # row). We grab the first one with [0] since we only sent one row. We wrap 
    # it in int() to turn the model's NumPy number into a plain 0 or 1, which 
    # behaves nicely when we send it back as JSON.
    prediction = int(rf_model.predict(df_processed)[0])

    # Finally, we send the answer back to whoever called us. We return two 
    # things: the raw number (easy for other programs to use) and a readable 
    # label (easy for a human to understand). If prediction is 1, it's "Fraud"; 
    # anything else is "Legit".
    return {
        "is_fraud": prediction,
        "label": "Fraud" if prediction == 1 else "Legit"
    }