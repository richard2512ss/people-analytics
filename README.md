# People Analytics — Employee Attrition Prediction

A logistic regression project that estimates the probability of an employee leaving
the company, built for HR decision support. Includes exploratory data analysis,
a fitted statistical model, and an interactive HTML simulator.

## Dataset

[IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Kaggle)

- 1,470 employees, 35 columns
- Fictional data created by IBM for demonstration purposes — not real company data
- Target variable: `Attrition` (`Yes` / `No`)

## Repository structure

```
people-analytics/
├── data/
│   └── hr_attrition.csv              Raw dataset (from Kaggle)
├── analysis/
│   ├── exploratory_analysis.Rmd      Full EDA in R (distributions, correlation, significance tests)
│   ├── variable_dictionary.Rmd       Description of every variable in the dataset
│   └── significance_ranking.csv      t-test / chi-square results, ranked by p-value
├── images/
│   ├── exploratory_overview.png      Summary chart: key variables vs attrition
│   ├── numeric_variables.png         Boxplots — continuous numeric variables
│   ├── ordinal_variables.png         Bar charts — ordinal/scale variables (1–4)
│   ├── categorical_variables.png     Bar charts — categorical variables
│   └── correlation_matrix.png        Correlation heatmap (numeric variables)
├── simulator/
│   └── attrition_simulator.html      Interactive probability calculator (standalone HTML)
└── README.md
```

## Exploratory data analysis

All 29 predictor variables were compared between employees who left and those who
stayed, using boxplots for continuous variables, bar charts for ordinal/categorical
variables, a correlation matrix, and statistical significance tests (Welch's t-test
for numeric variables, chi-square for categorical variables).

Key findings:

- **OverTime** is the single strongest predictor — employees who work overtime leave
  at roughly 3x the rate of those who don't.
- **MonthlyIncome**, **JobLevel**, **TotalWorkingYears**, **YearsAtCompany**, and
  **YearsWithCurrManager** are all strongly associated with attrition.
- **JobInvolvement** and **StockOptionLevel** show strong gradients — lower
  involvement / no stock options correlate with higher attrition.
- **PerformanceRating**, **Gender**, **Education**, **HourlyRate**, **DailyRate**,
  and **MonthlyRate** show no statistically significant relationship with attrition
  in this dataset.
- Several variables are highly correlated with each other (e.g. `JobLevel` and
  `MonthlyIncome`, r ≈ 0.95), which matters for model interpretation — see
  `analysis/exploratory_analysis.Rmd` for the full correlation matrix.

## Model

A logistic regression (`glm(..., family = binomial)`) was fitted in R to estimate
the probability of attrition from a set of statistically significant predictors:

```r
model <- glm(Attrition ~ Age + BusinessTravel + OverTime + MonthlyIncome +
               DistanceFromHome + YearsAtCompany + YearsSinceLastPromotion +
               JobInvolvement + WorkLifeBalance + EnvironmentSatisfaction +
               JobSatisfaction + NumCompaniesWorked,
             data = df, family = binomial)
summary(model)
```

All 12 predictors are statistically significant (p < 0.05). Notable effects
(odds ratios):

| Variable | Odds Ratio | Effect |
|---|---|---|
| OverTime = Yes | ~5.1x | Strongest single predictor |
| BusinessTravel = Frequently | ~5.2x | |
| JobInvolvement (per level) | ~0.56x | Higher involvement → lower risk |
| EnvironmentSatisfaction (per level) | ~0.69x | |
| MonthlyIncome (per $1,000) | ~0.90x | Higher income → lower risk |
| Age (per year) | ~0.95x | |

The model was validated against the full dataset: the average predicted
probability (16.1%) closely matches the actual attrition rate (16.1%), and
predicted probabilities are well separated between employees who left (avg. 36%)
and those who stayed (avg. 12%).

## Interactive simulator

`simulator/attrition_simulator.html` is a standalone, dependency-free HTML page
that lets you adjust each of the 12 model inputs with sliders and buttons, and see
the predicted probability of attrition update live. The probability is shown in a
sticky panel that stays visible while scrolling.

To use it, just open the file in any browser — no build step or server required.
It can also be published directly via GitHub Pages.

## Notes & limitations

- This dataset is synthetic and relatively small (1,470 rows), so results should
  be treated as illustrative of the methodology rather than as generalizable HR
  policy conclusions.
- `DailyRate`, `HourlyRate`, and `MonthlyRate` appear to be independently generated
  noise variables — they are not mathematically consistent with `MonthlyIncome` and
  showed no significant relationship with attrition.
- `PerformanceRating` only contains two values in this dataset (3 and 4), which
  limits its ability to discriminate between employees who left and stayed.
- Several predictors are collinear (e.g. income and job level); interpret
  individual coefficients with that in mind.

## Next steps

- [ ] Expand the simulator to cover all statistically significant variables
  (currently limited to the 12 used in the final model)
- [ ] Add a batch-scoring option (upload a CSV of employees, get risk scores for all)
- [ ] Validate the model on real, anonymized company data
- [ ] Consider testing tree-based models (Random Forest, XGBoost) with proper
  class-imbalance handling for comparison
