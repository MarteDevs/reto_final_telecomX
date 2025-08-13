# 🚀 TelecomX Customer Churn Analysis & Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776ab?style=for-the-badge&logo=python&logoColor=white)

**💀 BREAKING THE CHURN CYCLE WITH AI 💀**

*Data Science meets Machine Learning rebellion against customer loss*

[![License](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](#license)
[![GitHub stars](https://img.shields.io/github/stars/MarteDevs/reto_final_telecomX?style=flat-square&color=yellow)](#)
[![GitHub issues](https://img.shields.io/github/issues/MarteDevs/reto_final_telecomX?style=flat-square&color=orange)](#)

</div>

---

## 🎯 **MISSION STATEMENT**

> *"In a world where customers vanish like ghosts, we resurrect them with the dark arts of Machine Learning."*

This project is a complete end-to-end data science solution for **TelecomX**, analyzing customer churn patterns and building predictive models to identify at-risk customers. From raw data extraction to deployment-ready ML models, we weaponize every byte of information against customer loss.

---

## ⚡ **QUICK START**

```bash
# Clone the rebellion
git clone https://github.com/MarteDevs/reto_final_telecomX.git

# Enter the battlefield
cd reto_final_telecomX

# Install your arsenal
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn statsmodels

# Launch Phase 1: EDA Attack
jupyter notebook TelecomX_LATAM.ipynb

# Launch Phase 2: ML Warfare
jupyter notebook TelecomX_Maching_Learning.ipynb
```

---

## 🛠️ **TECH STACK**

| Tool | Purpose | Phase |
|------|---------|-------|
| 🐍 **Python** | Core language | Both |
| 🐼 **Pandas** | Data manipulation | Both |
| 🔢 **NumPy** | Numerical operations | Both |
| 📊 **Matplotlib** | Visualization engine | EDA |
| 🌊 **Seaborn** | Statistical plots | EDA |
| 🤖 **Scikit-learn** | Machine Learning | ML |
| ⚖️ **Imbalanced-learn** | SMOTE balancing | ML |
| 📈 **Statsmodels** | VIF analysis | ML |
| 📓 **Jupyter** | Interactive development | Both |

---

## 📊 **PROJECT ARCHITECTURE**

```
reto_final_telecomX/
├── 📓 TelecomX_LATAM.ipynb           # Phase 1: EDA & Data Analysis
├── 🤖 TelecomX_Maching_Learning.ipynb # Phase 2: ML Models & Prediction
├── 📄 README.md                     # Project documentation
├── 📊 visualizations/               # Generated plots
│   ├── churn_distribution.png
│   ├── contract_analysis.png
│   ├── correlation_heatmap.png
│   └── vif_analysis.png
└── 📑 reports/                     # Analysis insights
    ├── eda_report.md
    └── ml_performance.md
```

---

## 🔬 **DATA SCIENCE PIPELINE**

### **Phase 1: Exploratory Data Analysis** 🎣
**File**: `TelecomX_LATAM.ipynb`

- **🌐 Data Extraction**: JSON API consumption
- **🧹 Data Cleaning**: Null handling, duplicates removal
- **⚙️ Feature Engineering**: Created `Cuentas_Diarias` metric
- **📊 EDA**: Comprehensive statistical analysis
- **🎨 Visualization**: Contract, payment, tenure analysis

### **Phase 2: Machine Learning Pipeline** ⚙️
**File**: `TelecomX_Maching_Learning.ipynb`

- **🛠️ Data Preprocessing**: Feature encoding, normalization
- **🔍 Multicollinearity Analysis**: VIF computation and fixing
- **⚖️ Class Balancing**: SMOTE implementation
- **🤖 Model Training**: Logistic Regression & Random Forest
- **📈 Performance Evaluation**: ROC-AUC, confusion matrix, classification reports

---

## 💀 **KEY FINDINGS**

<div align="center">

### **THE CHURN BATTLEFIELD**

| Metric | Value | Impact |
|--------|-------|--------|
| 🟢 **Retained Customers** | 5,174 | 73.5% |
| 🔴 **Churned Customers** | 1,869 | 26.5% |
| 🎯 **Target Imbalance** | 3:1 ratio | SMOTE required |

</div>

### **🎯 HIGH-RISK PATTERNS**

- **📅 Month-to-Month Contracts**: 1,655 churned (88% of total churn)
- **💳 Electronic Check Payments**: 1,071 churned (57% of total churn)
- **🌐 Fiber Optic Users**: +0.31 correlation with churn
- **⏰ Early Tenure**: Median churn at 10 months vs 40 for retained

### **🛡️ RETENTION CHAMPIONS**

- **📋 Two-Year Contracts**: -0.30 correlation (loyalty factor)
- **🏦 Stable Payment Methods**: Lower churn rates
- **👥 Long-term Customers**: 60+ months show maximum loyalty
- **📞 No Internet Users**: -0.23 correlation (stable profiles)

---

## 🤖 **MACHINE LEARNING ARSENAL**

### **Data Preparation Weapons** 🔧

| Technique | Purpose | Implementation |
|-----------|---------|----------------|
| **One-Hot Encoding** | Categorical → Binary | `pd.get_dummies()` |
| **VIF Analysis** | Multicollinearity detection | `statsmodels.VIF` |
| **Standard Scaling** | Feature normalization | `StandardScaler()` |
| **SMOTE Balancing** | Class imbalance fix | `imbalanced-learn` |

### **Model Performance** 📊

| Model | Accuracy | ROC-AUC | Strengths |
|-------|----------|---------|-----------|
| **Logistic Regression** | TBD | TBD | Interpretable coefficients |
| **Random Forest** | TBD | TBD | Feature importance, robustness |

*Note: Run the ML notebook to see actual performance metrics*

---

## 📈 **VISUALIZATION GALLERY**

The project generates several punk-inspired data visualizations:

- **🎯 Churn Distribution Charts**: Class balance analysis
- **📋 Contract Type Breakdown**: Payment method impact
- **🌡️ Correlation Heatmaps**: Feature relationship analysis
- **⚡ VIF Analysis Plots**: Multicollinearity detection
- **🎭 Feature Importance**: ML model insights

---

## 🚨 **BUSINESS INTELLIGENCE**

### **IMMEDIATE ACTIONS** ⚡
1. **Monthly Contract Intervention**
   - Target customers at 6-12 month mark
   - Implement loyalty rewards program
   - Offer contract upgrade incentives

2. **Payment Method Optimization**
   - Investigate Electronic Check friction
   - Promote automatic payment methods
   - Provide payment support services

### **STRATEGIC WARFARE** 🎯
- **Predictive Churn Scoring**: Deploy ML models for real-time risk assessment
- **Customer Segmentation**: Personalized retention strategies
- **Proactive Outreach**: AI-driven customer success campaigns

---

## 🔧 **INSTALLATION & SETUP**

### **Prerequisites**
- Python 3.8 or higher
- Jupyter Notebook
- Internet connection (for data fetching)

### **Core Dependencies**
```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Advanced techniques
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

### **Installation Steps**
```bash
# Create virtual environment (recommended)
python -m venv telecom_env

# Activate environment
source telecom_env/bin/activate  # Linux/Mac
# or
telecom_env\Scripts\activate     # Windows

# Install all required packages
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn statsmodels jupyter
```

---

## 🎮 **USAGE GUIDE**

### **Phase 1: Data Exploration**
```bash
# Start with exploratory analysis
jupyter notebook TelecomX_LATAM.ipynb
```

**Key outputs:**
- Customer churn distribution
- Contract type analysis
- Payment method breakdown
- Correlation insights

### **Phase 2: Machine Learning**
```bash
# Build predictive models
jupyter notebook TelecomX_Maching_Learning.ipynb
```

**Key processes:**
- Data preprocessing pipeline
- Feature engineering and selection
- Model training and evaluation
- Performance comparison

---

## 🤝 **CONTRIBUTING**

Join the data science rebellion! Here's how to contribute:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/new-model`)
3. **Commit** your changes (`git commit -m 'Add XGBoost model'`)
4. **Push** to the branch (`git push origin feature/new-model`)
5. **Open** a Pull Request

### **Contribution Areas**
- 🤖 **New ML Models**: XGBoost, Neural Networks, Ensemble methods
- 📊 **Advanced Visualizations**: Interactive dashboards, 3D plots
- 🔍 **Feature Engineering**: New predictive variables
- 📈 **Performance Optimization**: Hyperparameter tuning
- 🧪 **A/B Testing**: Model comparison frameworks

---

## 📜 **LICENSE**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 **TEAM**

<div align="center">

**Data Science Rebel** 🎭  
*Turning customer data into retention gold*

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MarteDevs)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/martedevs)

</div>

---

## 🚀 **ROADMAP**

- [x] **Exploratory Data Analysis**: Complete customer behavior analysis
- [x] **Data Preprocessing**: Feature engineering and cleaning
- [x] **ML Pipeline**: Logistic Regression and Random Forest
- [ ] **Advanced Models**: XGBoost, Neural Networks, Ensemble methods
- [ ] **Real-time Scoring**: API deployment for live predictions
- [ ] **Interactive Dashboard**: Streamlit/Dash customer analytics
- [ ] **A/B Testing Framework**: Model comparison and validation
- [ ] **Customer Segmentation**: Advanced clustering analysis

---

<div align="center">

**💀 FIGHT THE CHURN. WIN WITH AI. 💀**

*Made with ❤️ and machine learning rebellion by the Data Science Underground*

---

**⭐ Star this repo if you found it useful! ⭐**

*"In data we trust, in models we predict, in customers we retain."*

</div>
