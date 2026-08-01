Customer Churn Prediction and Retention Targeting System
========================================================

Category: Machine Learning / Classification

Data Files: campaign_responses.csv, customer_churn.csv, customer_engagement_metrics.csv, customer_rfm.csv, customers.csv, orders.csv, subscription_billing.csv, support_tickets.csv


PROBLEM STATEMENT
-----------------

Integrated churn systems connect scoring, offer selection, and campaign list generation in one pipeline. Marketing ops needs repeatable exports keyed by customer_id with churn probability, segment, recommended offer, and expected value.
This project adds support ticket friction and full customer profiles to core churn tables for holistic targeting—high-risk customers with open escalated tickets receive different playbooks than silent churners.
Students deliver an end-to-end targeting system specification plus scored output validated against customer_churn.csv and campaign_responses.csv historical acceptance.


OBJECTIVES
----------

1. Integrate customer_churn.csv, customers.csv, customer_engagement_metrics.csv, and customer_rfm.csv.
2. Add support_tickets.csv friction features (open tickets, escalations) to churn models.
3. Join orders.csv for revenue-at-risk calculations on targeting lists.
4. Define offer selection rules using retention_offer_sent and offer_accepted history.
5. Export campaign-ready file: customer_id, churn_score, segment, offer_type, channel, expected_value.
6. Report system KPIs: precision@campaign size, acceptance rate, and revenue retained proxy.


NOTES
-----

Dataset Descriptions
~~~~~~~~~~~~~~~~~~~~
  - campaign_responses.csv: Marketing campaign exposure and response outcomes linked to customers and campaigns.
  - customer_churn.csv: Churn labels and behavioral features: tenure, spend, support contacts, NPS, and retention offer history.
  - customer_engagement_metrics.csv: Rolling engagement KPIs: logins, email opens, session counts, and feature adoption scores.
  - customer_rfm.csv: Recency, frequency, monetary values and R/F/M scores per customer for segmentation.
  - customers.csv: Customer master records with demographics, signup dates, contact attributes, and account metadata.
  - orders.csv: Order header transactions linking customers to purchase dates, amounts, and fulfillment status.
  - subscription_billing.csv: Subscription plan, billing cadence, payment failures, and contract type per customer.
  - support_tickets.csv: Customer support cases with category, status, subject, and created timestamps.

How Datasets Relate and Join
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Join customers.csv to orders.csv on customer_id for purchase history features.
- Join customer_churn.csv to customer_rfm.csv on customer_id for segmentation-aware churn modeling.

Suggested ML / Analytics Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Start with exploratory data analysis and baseline models (logistic regression or gradient boosting). Use stratified cross-validation, calibrate probabilities where decisions depend on thresholds, and report metrics aligned to business costs (false negative vs false positive tradeoffs).

Evaluation Metrics
~~~~~~~~~~~~~~~~~~
AUROC, AUPRC, precision-recall at top deciles, F1, and confusion matrix at operational thresholds.

Student Deliverables
~~~~~~~~~~~~~~~~~~~~
- Data exploration notebook or report documenting schema, missing values, and join diagrams for all project files.
- Implemented pipeline (Python preferred) reproducible from raw CSV/JSON/JSONL/DB files in this folder.
- Model, agent, or analytics outputs with held-out evaluation using the metrics above.
- Written summary (2-3 pages) interpreting results, limitations, and recommended production next steps.
- Artifact export appropriate to project type: scored CSV, recommendation lists, agent trace logs, dashboard screenshots, or generated report samples.

Technical Notes
~~~~~~~~~~~~~~~
- All data is synthetic and intended for education, portfolio demonstrations, and prototyping.
- Do not assume external APIs or live systems; simulate tool calls against local files.
- When using GenAI components, document prompts, retrieval configuration, and safety guardrails explicitly.
- Preserve reproducibility: set random seeds, document train/validation splits, and version any embedding models used.