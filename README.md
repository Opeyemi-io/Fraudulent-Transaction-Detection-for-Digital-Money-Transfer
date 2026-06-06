# Fraudulent-Transaction-Detection-for-Digital-Money-Transfer

### Business Challenge - Nova Pay Now Now
#### The Integrated Problem Statement
These four challenges are one problem seen from four angles. NovaPay needs a single capability that automatically scores every transaction for fraud risk, in real time, accurately enough to cut losses without harming customer experience, adaptably enough to track evolving tactics, and transparently enough to satisfy regulators in three markets.

#### The Business Problem: Four Core Challenges
1. The current human-and-rules approach can't match the platform's speed and scale. NovaPay clears cross-border transfers in seconds, which means fraud settles just as fast and the funds become unrecoverable, there is no window for after-the-fact review. Yet today's defenses are static rules and manual review: rules are rigid and fraudsters route around them, and human reviewers cannot keep pace with millions of real-time transactions. The defense simply cannot operate at the speed and volume the business runs at.
2. Fraud evolves, so any fixed solution decays. Fraudsters are active adversaries, they probe NovaPay's defenses and change tactics the moment they find a gap. A solution that catches today's fraud will catch steadily less of it within months, not because it broke, but because the fraud changed underneath it. The business needs a defense that can be monitored and refreshed over time, not a one-time fix.
3. Errors are costly in both directions, so the goal is balance, not zero fraud. Missing fraud triggers refunds, chargebacks, and regulatory penalties, direct, visible loss. Wrongly blocking a legitimate customer erodes trust and drives churn, a real but less visible loss. NovaPay could eliminate all fraud by blocking every transaction and destroy the company doing it, so the objective is not maximum fraud capture but an optimized balance between catching fraud and keeping genuine customers flowing smoothly.
4. Three markets mean three regulatory rulebooks. NovaPay operates in the UK, Canada, and the US, each with its own regulators and expectations on anti-money-laundering, consumer protection, and liability for unauthorized transfers. Whatever NovaPay builds must be not only accurate but explainable and defensible across all three jurisdictions, "the algorithm decided" is not an acceptable answer to a regulator.


### Identify the target variable for the prediction
 ##### Main Target variable = is fraud because it shows a skewed distribution of  fraudlent and legitmate transactions
 ##### Derived variables to be engineered from existing varibles to solve the business problem
-------
# 💳 Financial Transaction Fraud Detection

A Machine Learning system to identify fraudulent payment transactions. It helps **NovaPay** thwart identity theft, account takeover, and unauthorised transactions, protecting transfer integrity, earning customer and regulator trust, and keeping the user experience smooth.

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-LightGBM-success.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)

---

## 🎯 The Problem

NovaPay is a digital-first, cross-border money transfer company processing millions of transactions monthly across the UK, Canada, and US. Its current fraud defence relies on static rules plus manual review, which cannot scale or adapt fast enough to evolving fraud tactics.

The core challenges:

- 🕵️ **Evolving threats** like identity theft, account takeover, transaction laundering, and unauthorised payments.
- ⚖️ **Severe class imbalance**, with fraud making up less than 1% of all transactions.
- 😣 **False positives** that block legitimate users and drive customer attrition.
- 📜 **Regulatory pressure** for transparent, explainable, auditable decisions (AML / KYC).

## 💡 Why It Matters

- 🔒 **Platform integrity:** prevents financial loss and maintains trust with customers and regulators.
- ⚙️ **Operational performance:** lets fraud teams prioritise high-risk transactions and cut manual review load.
- 😀 **Customer satisfaction:** fewer false positives means fewer legitimate transactions wrongly blocked.
- 🏆 **Competitive edge:** ML-driven fraud detection is now standard among fintech leaders.

## 🚀 Objectives

1. Build supervised classifiers that accurately separate fraudulent from legitimate transactions across diverse patterns.
2. Handle the severe class imbalance through resampling and class-weighted ensemble methods.
3. Integrate **SHAP** explainability for transaction-level transparency for analysts and regulators.
4. Achieve at least a **15% improvement in recall** over the rules-based baseline while keeping precision acceptable.
5. Deploy as a **FastAPI microservice** for real-time scoring in production.

## 📊 Dataset

The data combines three categories tied to a clear structure (Transaction → Customer is many-to-one; Transaction → Fraud label is one-to-one).

| Category | What it captures |
|---|---|
| 🧾 **Transaction Data** | IDs, amounts, currency pairs, timestamps, channel (mobile / web / ATM), device fingerprints, IP addresses, country codes |
| 👤 **Customer Data** | Account age, KYC tier, typical amounts, historical behaviour, internal risk scores |
| 🏷️ **Fraud Labels** | Binary 0/1 ground truth from completed investigations, confirmed chargebacks, and verified disputes |

## 🔁 Workflow

| Step | Stage | Focus |
|---|---|---|
| 1️⃣ | **Data Collection & Profiling** | Distributions, missingness, baseline fraud prevalence |
| 2️⃣ | **Data Preparation** | Cleaning, feature engineering (velocity, mismatch indicators, device fingerprints, temporal patterns) |
| 3️⃣ | **Exploratory Data Analysis** | Patterns by channel, geography, and time; anomaly detection |
| 4️⃣ | **Model Development** | Logistic Regression and Random Forest baselines, then XGBoost and LightGBM; hyperparameter tuning |
| 5️⃣ | **Validation & Explainability** | Holdout evaluation plus SHAP for transaction-level reasoning |
| 6️⃣ | **Deployment** | FastAPI `/score` endpoint, containerised with Docker |
| 7️⃣ | **Monitoring & Improvement** | Evidently drift detection and a quarterly retraining playbook |

**Evaluation metrics:** Recall, Precision, F1-score, and ROC-AUC, with recall prioritised for catching fraud.

## 🧰 Tech Stack

- **Language:** Python
- **Modelling:** scikit-learn, XGBoost, LightGBM
- **Explainability:** SHAP
- **Serving:** FastAPI
- **Containerisation:** Docker
- **Monitoring:** Evidently

## 📦 Deployment

The trained model is served as a FastAPI microservice exposing a `/score` endpoint for real-time transaction evaluation, containerised with Docker for portable deployment across environments. Drift monitoring with Evidently and a quarterly retraining cycle keep the model accurate as fraud patterns shift.

## 🏁 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection

# Set up the environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the API locally
uvicorn app.main:app --reload
```

## 📄 License

Released under the **MIT License**. See [LICENSE](LICENSE) for details.