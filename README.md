# House Price Prediction: A Multiple Regression & Ensemble Modeling Study

## 1. Project Title & Overview

This project builds and compares statistical models to predict residential **house sale prices** from ~80 structural, locational, and quality-related property features (lot size, living area, overall quality, neighborhood, basement/garage attributes, etc.). It addresses the classic real-estate valuation problem — estimating a fair market price from observable property characteristics — by taking a rigorous statistics-first path: extensive EDA and normality diagnostics, systematic missing-data recovery, multicollinearity remediation via VIF, AIC-based stepwise variable selection, and outlier removal via residual/Cook's-distance analysis, culminating in a final interpretable multiple linear regression model. The same cleaned feature set is then benchmarked against three tree-based ensemble methods (Bagging, Random Forest, XGBoost) to quantify how much predictive accuracy a fully interpretable linear model gives up relative to black-box alternatives — valuable for use cases (e.g., appraisal, underwriting) where explainability of the price driver coefficients matters as much as raw accuracy.

## 2. Core Methodologies & Architectural Approach

**Response variable normalization.** EDA on `SalePrice` showed a strongly right-skewed histogram, attributed in the analysis to limited demand (fewer wealthy buyers) at the high end of the market. Since "for modelling we'll like the data to be normally distributed so that the skewed values don't impact our multiple linear regression model," a `log(SalePrice)` transform was applied and used as the actual modeling target (`SalePricelog`) throughout.

**Missing data recovery based on the data dictionary, not blind imputation.** Investigation of each variable's codebook revealed that most `NA` values were not truly missing — they encoded the *absence* of a feature (e.g., no garage, no pool). 18 variables were explicitly recoded this way (`GarageType`→`"NoGarage"`, `PoolQC`→`"NoPool"`, `BsmtQual`→`"NoBasement"`, `LotFrontage`→`0`, `MasVnrArea`→`0`, `MasVnrType`→`"None"`, etc.). Only **one row** of genuinely missing data remained after this pass and was dropped via `complete.cases()`.

**Type correction for downstream modeling.** 43 character-typed columns were cast to `factor` so they could be used correctly in EDA visualizations and in `lm()`/tree-based training.

**Feature engineering around class imbalance.** Several categorical predictors were dominated by one level (e.g., `Neighborhood` has 25 distinct levels but only a handful drive most of the price variance). The stated rule: any sub-category representing less than 5% of samples was a recategorization candidate, grouped by "aligning the features with some commonality on the house sale price." Examples implemented: `Neighborhood` → `nbrstatus` (High: StoneBr/NridgHt/NoRidge; Low: BrDale/IDOTRR/MeadowV; Mid: everything else), `LotShape` → Regular vs. Irregular, `MSZoning` → RL/RMH/OTH, `Functional` → `Functional1` (Typ vs. Dmg), plus similarly regrouped versions of `Condition1/2`, `Exterior1st/2nd`, `BldgType`, `RoofStyle`, `MasVnrType`, `Electrical`, `Fence`, `LandContour`, `LotConfig`, `LandSlope`, `SaleType`, `HouseStyle`, and `PavedDrive`. Variables where over 95% of samples fell into a single category with no compensating pattern (`Street`, `Utilities`, `PoolQC`, `RoofMatl`, `Heating`, `MiscFeature`) were dropped outright as uninformative; the correlated `HeatingQC` was dropped alongside `Heating` for the same reason.

**Normality-driven transformation via Box-Cox.** All numeric predictors were screened with a Box-Cox procedure to test whether a transform was needed. Only `LotArea` and `GrLivArea` required one; since their estimated lambdas were close to zero, a `log()` transform was applied to both (`LotArealog`, `GrLivArealog`) rather than a general Box-Cox power transform.

**Interaction/composite feature construction.** Two engineered variables were added: `Remodeled` (categorical yes/no, `YearBuilt != YearRemodAdd`) and `Bath`, a single continuous bathroom-count feature combining all four bathroom-type columns: `Bath = FullBath + 0.5·HalfBath + BsmtFullBath + 0.5·BsmtHalfBath`.

**Multicollinearity remediation via correlation analysis and VIF.** After an initial AIC-selected model threw a `vif()` error ("aliased coefficients," traced to `BsmtFinType1`), a VIF audit of the corrected model surfaced widespread VIF > 10. Rather than mechanically dropping variables, redundant metrics were consolidated using near-perfect correlations confirmed in code: `Bath` (r = 1 with its component sum), `GrLivArea` (r = 1 with `X1stFlrSF + X2ndFlrSF + LowQualFinSF`), and `TotalBsmtSF` (r = 1 with the sum of its component basement-area fields). The model was then re-specified around the aggregate variables only, dropping their now-redundant components (individual bathroom counts, floor-area splits, basement sub-areas, and the basement-quality-per-floor variables whose underlying areas had been removed). `GarageCars` was kept over the highly correlated `GarageArea` because it had a higher raw correlation with `SalePrice` and is described as "more understandable for buyers" — an explicit interpretability-driven choice, not just a statistical one.

**Model selection strategy: stepwise AIC in three directions.** `stepAIC()` was run in `forward`, `backward`, and `both` (bidirectional) modes on a full model (`SalePricelog ~ . - Id`). Backward and bidirectional search converged to the same ~41-variable model; forward search failed to meaningfully reduce the original ~74–80-variable set, which the authors call out explicitly as a limitation of forward selection here.

**Outlier removal via residual and Cook's-distance diagnostics.** Standard `plot(lm_model)` diagnostic panels (residuals vs. fitted, Q-Q, scale-location, residuals vs. leverage/Cook's distance) were inspected on the multicollinearity-corrected model. Two points (row indices 524 and 1299, both with `GrLivArea > 4500` but anomalously low sale prices — one tied to houses in "deteriorated condition" in low-residential areas, one sold during the housing crisis) were identified as high-leverage outliers and removed before the final refit.

**Significance-driven final pruning.** After retraining post-outlier-removal, non-significant predictors by p-value (`MiscVal`, `Condition2M`, `LotConfigM`, `RoofStyleM`, `ExterCondM`, `HouseStyleM`) were dropped to arrive at the final specification, `mfinal_reg`.

**Train/test protocol.** A fixed 80/20 random split (`set.seed(123)`) was used throughout — 80% for training, 20% held out to check for overfitting by comparing train vs. test R².

**Benchmarking against tree ensembles.** For the Bagging, Random Forest, and XGBoost models, the log-transformed and engineered continuous variables (`LotArealog`, `GrLivArealog`) were explicitly *not* used — the narrative states "we are not using any of the transformation methods on the variables" for the tree ensemble path, reflecting that tree splits don't require normalized or log-transformed inputs the way OLS does. All three ensemble models used 5-fold, 2-repeat repeated cross-validation (`trainControl(method='repeatedcv', number=5, repeats=2)`) via `caret::train()`.

## 3. Models & Algorithms Deployed

| # | Model | Role | Key Configuration | Why Used (per the document) |
|---|---|---|---|---|
| 1 | **Stepwise AIC Linear Regression (forward/backward/both)** | Automated variable-selection pass over the full ~74–80-variable model | `MASS::stepAIC()` on `lm(SalePricelog ~ . - Id)`, all three `direction` modes compared | To let AIC drive an initial, defensible reduction in the very large categorical/numeric feature space before manual multicollinearity work |
| 2 | **OLS Multiple Linear Regression — intermediate (`m2`, `m2_mod`, `m3`)** | Diagnostic/iterative models used to isolate and fix multicollinearity and aliasing | Built from the backward/both AIC variable set, refit repeatedly while removing aliased (`BsmtFinType1`) and high-VIF/redundant predictors | To trace VIF > 10 issues to specific redundant feature groups (bathrooms, floor areas, basement areas) and correct them one issue at a time |
| 3 | **OLS Multiple Linear Regression — final (`mfinal_reg`)** | Primary interpretable model / final regression deliverable | `lm(SalePricelog ~ OverallQual + OverallCond + YearRemodAdd + BsmtExposure + TotalBsmtSF + CentralAir + Bath + KitchenAbvGr + Fireplaces + GarageCars + PavedDriveM + SaleCondition + Condition1M + Exterior1M + nbrstatus + Functional1 + Remodeled + MSZoningM + LotArealog + GrLivArealog)`, fit on data with outliers (rows 524, 1299) removed | Chosen specification after VIF remediation, outlier removal, and p-value-based pruning; coefficients are described as directly interpretable (e.g., neighborhood-status coefficients scale with market tier, bathroom count preferred over bedroom count because it produced lower p-values and better R²) |
| 4 | **Bagging (Bootstrap Aggregation)** | First tree ensemble benchmark | `caret::train(method='treebag')`, `repeatedcv` (5-fold, 2 repeats), variable importance extracted (top 20, 15 of which have >10% importance) | First non-linear benchmark against the regression model, trained on untransformed features |
| 5 | **Random Forest** | Second tree ensemble benchmark | `caret::train(method='rf')`, same repeated-CV control, variable importance extracted (top 20, 15 with >10% importance) | Compared against Bagging to see whether decorrelated trees improve on simple bootstrap aggregation |
| 6 | **XGBoost (Extreme Gradient Boosting)** | Best-performing benchmark model | `caret::train(method='xgbTree')`, fixed grid (`nrounds=500, max_depth=3, eta=0.2, gamma=2.1, colsample_bytree=1, min_child_weight=1, subsample=1`) | Described as sampling data based on high-error terms iteratively; reported as the best-performing model of the four compared |

**Model comparison conclusion (as stated in the document):** the simple linear regression model is easiest to interpret because of its explicit coefficient weights; the tree ensembles overall outperformed the multiple regression model; XGBoost performed best among all four; the tree ensembles showed mild overfitting, judged to be "under threshold" and fixable with more data over time.

## 4. Tech Stack & Libraries

This is an **R / RStudio / R Markdown** project (not Python) — analysis, modeling, and narrative are combined into a single `.Rmd` document that knits to Word/HTML.

- **Core language & authoring:** R, R Markdown (`rmarkdown`), knit targets `word_document` and `html_document`
- **Data wrangling:** `dplyr`, `readr`, `datasets`
- **Visualization / EDA:** `ggplot2`, `GGally` (`pairs.panels`-style pair plots), `corrplot`, `psych` (correlation/pair panels), `visdat`, `naniar` (missing-data visualization utilities, though largely commented out in favor of manual `complete.cases()` checks)
- **Missing data:** `mice` (imported; primary missing-value strategy actually used was manual domain-driven recoding rather than multiple imputation)
- **Statistical modeling (linear regression, stepwise selection, diagnostics):** `MASS` (`stepAIC`), base R `lm`/`aov`/`anova`, `car` (`vif`, `alias`)
- **Classical stats / hypothesis testing utilities:** `BSDA`, `DescTools`, `DAAG`
- **Classification / general ML utilities (imported, supporting the ensemble section):** `nnet`, `e1071`, `pROC`
- **Ensemble modeling & cross-validation harness:** `caret` (`train`, `trainControl`, `varImp`), tree/bagging/RF/XGBoost engines (`treebag`, `rf`, `xgbTree`)
- **Model interpretability (imported):** `lime`
- **Multivariate/exploratory support:** `mlbench`
- **Dataset:** `HousingDatatrain.csv`, loaded via `read.csv('HousingDatatrain.csv')`. Based on the variable names used throughout (`SalePrice`, `Neighborhood`, `OverallQual`, `GrLivArea`, `MSZoning`, `Id`, ~80 predictors), this matches the structure of the well-known **Ames, Iowa housing dataset** (as distributed in Kaggle's "House Prices: Advanced Regression Techniques" competition) — the .Rmd itself does not explicitly cite a source URL, so this identification is inferred from the schema, not stated in the document.

## 5. Key Results & Engineering Metrics

The `.Rmd` source contains R code that computes model performance (`summary()`, `predict()`, `cor()^2`, RMSE calculations, `print(paste(...))`), but because this is unrendered source (not a knitted/executed notebook), most of that code's printed output is not captured inline in the text — only the results the authors explicitly wrote out in prose or left as commented output are available here. Fabricated numbers are avoided; anything not below simply wasn't present in the source.

**Stepwise AIC variable-selection comparison (explicitly stated in the narrative, as percentages):**

| Selection direction | Variables retained | Train R² | Adj. R² | Test R² |
|---|---|---|---|---|
| Forward | 74 | 93.13 | 92.11 | 81.57 |
| Backward / Both | 41 | 92.7 | 92.15 | 82.29 |

A second, slightly different pass of the same comparison is also stated in the text:

| Selection direction | Train R² | Test R² | Variable count |
|---|---|---|---|
| Both / Backward | 92.02 | 81.40 | 42 |
| Forward | 91.93 | 81.05 | 72 |

The document's own reading of these numbers: **all candidate AIC models are overfitting** (training R² consistently ~10 points above test R²); the backward/bidirectional model achieves comparable or better test performance with roughly half the variables of the forward model, and forward selection "failed to reduce the number of variables" from the original ~80.

**Multicollinearity:** the initial AIC-selected model contained an aliased coefficient (`BsmtFinType1NoBsmtFinType1`) and, once corrected, showed multiple predictors with VIF > 10. After consolidating redundant area/bathroom/basement metrics into aggregate variables (`Bath`, `GrLivArea`, `TotalBsmtSF`, `GarageCars`), the document states VIF values fell below the conventional threshold of 10.

**Outliers:** two observations (row indices **524** and **1299**) were identified via Cook's-distance/residual diagnostics as high-leverage, anomalously low-priced, large-`GrLivArea` (>4500 sq ft) houses and removed prior to the final model fit.

**Final OLS model (`mfinal_reg`) and the Bagging/Random-Forest R²/RMSE:** the source computes these via `print(paste("R-square_training=", r_reg_train))`-style statements, but the .Rmd does not embed the executed numeric output for these particular calls — no train/test R² or RMSE values for the final linear model, Bagging, or Random Forest are hard-coded in the text, so none are reported here to avoid fabrication.

**XGBoost — the only ensemble model with numeric output actually recorded in the source** (left as a commented-out result block in the code chunk):

| Metric | Value |
|---|---|
| R² (train) | 0.9992 |
| R² (test) | 0.9419 |
| RMSE (train) | 3,146.09 |
| RMSE (test) | 25,025.91 |

**Overall conclusion stated in the document:** tree ensembles outperformed the multiple linear regression model overall, XGBoost was the best performer of the four models compared, the linear model remains preferable for interpretability of individual coefficient effects, and the ensemble models' mild overfitting was judged acceptable and likely improvable with more data.

## 6. Repository Structure

```
stat_E109 - House Pricing Prediction/
├── Project_work_Final_V1.0.Rmd     # Primary analysis: EDA, missing-data handling, feature
│                                    #   engineering, transformation, stepwise AIC selection,
│                                    #   VIF/multicollinearity remediation, outlier removal,
│                                    #   final OLS regression, and Bagging/RF/XGBoost benchmarks
├── Project_work_Final_V5.docx      # Supplementary rendered project report (Word document,
│                                    #   ~2.8 MB) — the formatted write-up counterpart to the
│                                    #   .Rmd source; not parsed for this README (binary format)
├── README.md                       # This file
└── HousingDatatrain.csv            # NOT included in this repository — required at runtime,
                                     #   loaded via read.csv('HousingDatatrain.csv')
```

## 7. Quickstart & Usage

**1. Install R dependencies** (run once in an R/RStudio console; no `renv`/`packrat` lockfile or `DESCRIPTION` is committed, so install every package the document `library()`-loads):

```r
install.packages(c(
  "DAAG", "ggplot2", "dplyr", "psych", "caret", "pROC", "nnet",
  "e1071", "car", "MASS", "mlbench", "BSDA", "readr", "visdat",
  "naniar", "GGally", "mice", "lime", "DescTools", "corrplot",
  "rmarkdown", "xgboost"
))
```

(`datasets` is bundled with base R and needs no install; `xgboost` is required indirectly by `caret::train(method='xgbTree')`.)

**2. Provide the dataset.** Place `HousingDatatrain.csv` in the same directory as the `.Rmd` file — the document reads it via:

```r
df_housing = read.csv('HousingDatatrain.csv')
```

Given the variable schema used throughout (SalePrice, OverallQual, GrLivArea, Neighborhood, ~80 predictors, an `Id` column), a compatible file is the Ames Housing training set distributed with Kaggle's "House Prices: Advanced Regression Techniques" competition; this is an inference from the code, not a citation present in the document itself.

**3. Knit or run the analysis.** From an R console in the project directory:

```r
rmarkdown::render("Project_work_Final_V1.0.Rmd")
```

This reproduces the full pipeline end-to-end (per the `output:` front-matter, defaulting to a Word document, with an HTML option available) — EDA plots → missing-data cleanup → variable-type conversion → feature engineering/recategorization → Box-Cox-informed log transforms → 80/20 split → stepwise AIC selection → VIF/multicollinearity correction → Cook's-distance outlier removal → final regression model → Bagging/Random Forest/XGBoost ensemble comparison. Alternatively, open the file in RStudio and run chunks sequentially (`Run All`) to inspect intermediate plots and model summaries interactively.
