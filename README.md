# Predicting Patient Disengagement in Remote Therapeutic Monitoring

A data science capstone project conducted in partnership with **Flagler Health**, analyzing clinical, administrative, and patient-communication data to identify factors associated with patient disengagement from a Remote Therapeutic Monitoring (RTM) program.

The project was awarded **2nd Place at Carnegie Mellon University's Meeting of the Minds**.

---

## Project Overview

Flagler Health provides Remote Therapeutic Monitoring services for patients with musculoskeletal conditions. Patient retention is important both for continuity of care and for ongoing monitoring.

Our team analyzed data from **391 patients** enrolled between August and December 2025 to answer:

> **What factors can we use to predict patient disengagement?**

The project combined structured clinical and insurance data with unstructured patient–Medical Assistant conversations.

### Methods

- Data cleaning and integration with SQL / DuckDB
- CPT procedure and ICD-10 diagnosis categorization
- Exploratory and longitudinal data analysis
- Kaplan-Meier survival analysis
- Cox proportional hazards modeling
- Logistic regression
- NLP sentiment analysis using Flair
- Model comparison using AIC, ANOVA, and Hosmer-Lemeshow testing

---

## Data

The analysis combined four sources:

| Data Source | Examples |
|---|---|
| Patient Metrics | Pain, mobility, mood, sleep |
| Status Records | Enrollment and disenrollment |
| Insurance Claims | CPT procedure and ICD-10 diagnosis codes |
| Messages | Patient–Medical Assistant conversations |

The original patient-level data are not included in this repository.

---

## Analytical Approach

### 1. Exploratory Analysis

We examined longitudinal patient metrics and compared engagement patterns between patients who remained enrolled and those who disengaged.

### 2. Survival Analysis

Kaplan-Meier curves and Cox proportional hazards models were used to analyze both whether and when patients disengaged.

Baseline predictors included:

- Pain
- Mobility
- Mood
- Sleep
- Procedure group
- Diagnosis group

### 3. NLP Sentiment Analysis

Patient messages contained information that was not captured by the structured variables alone.

Using the **Flair NLP library**, message sentiment was converted into a continuous score ranging from negative to positive sentiment and incorporated into the final logistic regression model.

### 4. Model Evaluation

We compared models with and without sentiment using:

- ANOVA model comparison
- Akaike Information Criterion (AIC)
- Hosmer-Lemeshow goodness-of-fit testing

---

## Key Findings

### Patient sentiment was the strongest predictor of disengagement

The final logistic regression estimated:

**Sentiment coefficient = -1.29 (p = 0.01)** corresponding to an odds ratio of approximately **0.28**.

A one-unit increase in average sentiment was therefore associated with approximately **72% lower odds of patient dropout**, holding the other modeled variables constant.

Adding sentiment also improved model fit:

| Model Evaluation | Result |
|---|---|
| AIC with sentiment | 294.02 |
| AIC without sentiment | 299.36 |
| ANOVA comparison | p = 0.00673 |
| Hosmer-Lemeshow | p = 0.72 |

### Procedure type also showed meaningful retention patterns

Procedure grouping produced the clearest differences in dropout patterns among the structured variables and showed similar directions of association across Cox and logistic regression models.

Baseline pain, mobility, mood, and sleep alone provided relatively limited predictive information.

---

## My Contributions

This project was completed by a four-person team.

My contributions included:

- Exploratory analysis of longitudinal patient data
- Survival analysis using Kaplan-Meier curves and Cox models
- Logistic regression modeling
- Implementing NLP sentiment analysis using Flair
- Integrating sentiment into the final logistic regression model
- Evaluating model improvement using AIC, ANOVA, and Hosmer-Lemeshow testing
- Contributing to the final report, presentation, and poster

---

## Tools

**Python:** pandas, NumPy, Flair  
**R:** dplyr, ggplot2, survival, broom  
**SQL:** DuckDB  
**Methods:** Logistic Regression, Survival Analysis, NLP, Feature Engineering, Model Evaluation

---

## Repository Structure

```text
data/
    README.md
src/
    01_exploratory_analysis.Rmd
    02_survival_and_logistic_models.Rmd
    03_sentiment_analysis.py

Final_Presentation.pdf
Final_Report.pdf
Project_Poster.png
README.md
requirements.txt
