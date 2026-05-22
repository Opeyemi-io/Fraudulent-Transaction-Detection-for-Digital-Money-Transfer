# NovaPay Dataset, Data Quality Assessment

## Summary
### The dataset is usable but requires cleaning and careful feature validation before modeling. Priority actions: remove duplicate rows, standardize categorical values, correct the amount_src data type, unify the treatment of missing values, adopt precision-recall-based evaluation in light of the 8.75% fraud rate, and confirm whether chargeback_history_count and risk_score_internal are free of label leakage. The raw file should be preserved unchanged, with all cleaning applied to a separate working copy.

## Doumentation details
The dataset contains 11,400 transaction records across 26 columns. Exploration identified seven findings spanning data quality, class balance, and label reliability. Each must be addressed before modeling, as each has the potential to distort predictions.

1. Duplicate records
200 fully duplicated rows were identified (1.75% of the dataset). The duplication is confirmed at the transaction_id level, repeated IDs with identical row content, indicating the same transaction was recorded twice, likely from combining source files. The true unique transaction count is therefore 11,200. These rows should be removed during cleaning so each transaction is counted once.

2. Missing values
Seven columns contain missing values:
timestamp, 29 missing
amount_usd, 305 missing
fee, 295 missing
ip_address, 305 missing
ip_country, 301 missing
kyc_tier, 300 missing
device_trust_score, 295 missing
All other columns, including the target variable is_fraud, are complete. The missing values will need to be handled during cleaning, either through imputation or, where appropriate, exclusion.

3. Incorrect data type
The amount_src column loaded as text (string) when it should be numeric. This indicates non-numeric or malformed values are mixed into the column. It must be cleaned and converted to a numeric type before it can be used in analysis or modeling.

4. Inconsistent categorical values
Several categorical columns contain the same real category recorded multiple different ways, caused by inconsistent capitalization, stray whitespace, and typos:
channel has 12 distinct values representing only 3 true channels (ATM, web, mobile), including typos such as weeb and mobille.
kyc_tier has 15 distinct values for 3 true tiers (standard, enhanced, low), including typos such as standrd and enhancd.
home_country has 7 values for 3 countries (US, CA, UK), caused mainly by stray whitespace.
source_currency is clean (USD, CAD, GBP) and requires no correction.
Because a model treats each distinct string as a separate category, these inconsistencies fragment the data and weaken predictive patterns. They require standardization during cleaning, trimming whitespace, unifying case, and correcting typos.

5. Inconsistently recorded missing values
In kyc_tier, missing values appear in multiple forms (nan, NAN, and nan), and an unknown category is also present. These need to be unified into a single, consistent representation of "missing" during cleaning.

6. Class imbalance
The target variable is_fraud is imbalanced. Of the 11,400 records, 10,403 are legitimate (91.25%) and 997 are fraud (8.75%). Fraud is the minority class. This has a direct consequence for modeling: accuracy is not a valid evaluation metric, since a model predicting "not fraud" for every transaction would score over 91% accuracy while catching no fraud. Evaluation must instead use precision, recall, and PR-AUC, and the imbalance should be accounted for during model training.

7. Potential label leakage, chargeback_history_count
A crosstab of chargeback_history_count against is_fraud shows an extremely strong relationship: when the count is 0, fraud rate is low (580 of 10,958); when the count is 1, transactions are predominantly fraud (307 of 331); when the count is 2, they are almost entirely fraud (110 of 111). Overall, when chargeback_history_count is greater than 0, roughly 94% of transactions are fraud.
This relationship is strong enough to raise a leakage concern. The feature is only valid if it counts chargebacks from the customer's prior transactions. If it includes a chargeback tied to the current transaction, it partly encodes the outcome itself, which would produce a model that performs artificially well in testing but fails in production. The construction of this column must be confirmed against dataset documentation before it is used as a feature. risk_score_internal should be reviewed for the same reason, as an internally computed score may have been derived after fraud confirmation.