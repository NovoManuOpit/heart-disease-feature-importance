# Heart Disease Feature Importance Analysis

Investigating which medical features contribute most to predicting heart disease using four machine learning classifiers and six complementary feature importance methods.

## Research Question

> Do certain medical features contribute significantly more to heart disease prediction than others?

**Answer: Yes.** Across all models and methods, a small group of features carries the overwhelming share of predictive signal while several features contribute near zero.

## Key Findings

### Top Predictive Features

| Rank | Feature | Avg. Importance | Consistent Across |
|:----:|---------|:---------------:|:-----------------:|
| 1 | Chest Pain Type | 1.312 | All 6 methods |
| 2 | Num Major Vessels | 1.299 | All 6 methods |
| 3 | Thalassemia Test Results | 0.976 | 5 / 6 methods |
| 4 | ST Depression | 0.506 | All 6 methods |
| 5 | Max Heart Rate | 0.359 | 4 / 6 methods |
| 6 | Age | 0.338 | 4 / 6 methods |

**Low-value features:** Resting ECG, Cholesterol, and Fasting Blood Sugar score consistently near zero across all methods.

### Model Performance

| Model | Accuracy | F1 Score |
|-------|:--------:|:--------:|
| Logistic Regression | 87.3% | 0.880 |
| Decision Tree (depth=5) | 88.8% | 0.892 |
| Random Forest (200 trees) | 100%* | 1.000* |
| Gradient Boosting (200 trees) | 100%* | 1.000* |

> \*Perfect scores are an artefact of duplicate rows in the merged UCI dataset. Logistic Regression and the depth-limited Decision Tree are the more honest indicators of true generalisation (~87-89%).

## Dataset

- **Source:** [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) (merged from multiple UCI repositories)
- **Samples:** 1,025 rows
- **Features:** 13 clinical attributes + 1 binary target
- **Classes:** Balanced (~51% disease, ~49% no disease)

### Features

| Type | Features |
|------|----------|
| Numeric | Age, Resting Blood Pressure, Cholesterol, Max Heart Rate, ST Depression |
| Binary | Sex, Fasting Blood Sugar, Exercise Angina |
| Categorical | Chest Pain Type, Resting ECG, ST Slope, Thalassemia Test Results, Num Major Vessels |

## Methodology

### Preprocessing
- No missing values in dataset
- One-hot encoding for categorical features (13 raw &rarr; 27 encoded)
- StandardScaler on numeric features (fit on train only &mdash; no data leakage)
- Stratified 80/20 train/test split (seed=42)

### Models
- **Logistic Regression** &mdash; linear baseline
- **Decision Tree** (max_depth=5) &mdash; interpretable, depth-limited
- **Random Forest** (200 trees) &mdash; ensemble, bagging
- **Gradient Boosting** (200 trees, lr=0.1) &mdash; ensemble, boosting

### Feature Importance Methods (6 total)

| # | Method | Type |
|:-:|--------|------|
| 1 | Logistic Regression coefficients | Native (absolute values) |
| 2 | Decision Tree MDI | Native (Gini impurity) |
| 3 | Random Forest MDI | Native (averaged over trees) |
| 4 | Gradient Boosting MDI | Native (across boosting sequence) |
| 5 | SHAP values (TreeExplainer) | Game-theoretic, per-sample directional |
| 6 | Permutation importance (F1, 30 repeats) | Model-agnostic, generalisation-based |

## Sample Visualizations

The analysis generates 11 plots saved to `notebooks/outputs/`:

| File | Description |
|------|-------------|
| `model_comparison.png` | Accuracy & F1 across all four models |
| `confusion_matrices.png` | Side-by-side confusion matrices |
| `feature_importance_per_model.png` | Top 15 native features per model |
| `cross_model_importance.png` | Grouped bar chart &mdash; top 10 features by model |
| `avg_importance.png` | Average importance across all models |
| `shap_beeswarm.png` | Per-sample SHAP values with feature direction |
| `shap_bar.png` | Mean absolute SHAP values |
| `permutation_importance.png` | Permutation importance with error bars |
| `summary_heatmap.png` | All methods x all features heatmap |
| `final_feature_ranking.png` | Final ranking averaged across all 6 methods |

## Project Structure

```
heart-disease-feature-importance/
├── README.md
├── notebooks/
│   ├── Preprocessing_EDA_.ipynb          # Data loading, EDA, preprocessing
│   ├── Modeling_Feature_Importance.ipynb  # Model training & importance analysis
│   └── outputs/
│       ├── preprocessed/
│       │   ├── train.csv                 # 820 samples x 28 columns
│       │   └── test.csv                  # 205 samples x 28 columns
│       ├── model_comparison.png
│       ├── confusion_matrices.png
│       ├── feature_importance_per_model.png
│       ├── cross_model_importance.png
│       ├── avg_importance.png
│       ├── shap_bar.png
│       ├── shap_beeswarm.png
│       ├── permutation_importance.png
│       ├── summary_heatmap.png
│       └── final_feature_ranking.png
```

## How to Run

### Prerequisites

- Python 3.9+
- Kaggle API credentials (for dataset download)

### Setup

```bash
git clone https://github.com/NovoManuOpit/heart-disease-feature-importance.git
cd heart-disease-feature-importance
pip install pandas numpy matplotlib seaborn scikit-learn shap kagglehub scipy
```

### Execution

Run the notebooks in order:

1. **`Preprocessing_EDA_.ipynb`** &mdash; downloads the dataset, performs EDA, exports preprocessed train/test CSVs
2. **`Modeling_Feature_Importance.ipynb`** &mdash; trains models, runs all importance methods, generates plots

## Known Limitations

- **Duplicate rows:** The Kaggle dataset merges multiple UCI sources, causing identical rows in train and test sets. Tree ensembles memorise these, inflating their scores.
- **Single dataset:** Rankings cannot be generalised without validation on an independent cohort.
- **MDI bias:** Mean Decrease in Impurity overestimates features with many unique values or many one-hot dummies. SHAP and permutation importance are included as cross-checks.
- **Binary target:** The original UCI data encodes five severity levels (0-4). Collapsing to binary discards clinically meaningful gradations.

## Technologies

- **Data:** pandas, NumPy, SciPy
- **ML:** scikit-learn
- **Explainability:** SHAP (TreeExplainer), permutation importance
- **Visualization:** Matplotlib, Seaborn
- **Data Source:** kagglehub
