# 🧠 MindScope: Mental Health & Lifestyle Predictor

## 📌 Overview

MindScope is an end-to-end Machine Learning project that analyzes lifestyle and behavioral patterns to predict mental health risk levels using survey data.

The project follows a complete Data Science pipeline starting from raw data cleaning all the way to deployment through a GUI application.

---

# 🎯 Project Goal

Build a machine learning system capable of predicting mental health risk based on lifestyle habits such as:

* Sleep duration
* Screen time
* Physical activity
* Stress levels
* Daily habits

---

# 📊 Dataset

**Dataset:** Global Mental Health & Lifestyle Survey

### Dataset Characteristics

* Tabular dataset
* Numerical + categorical features
* Contains:

  * Missing values
  * Inconsistent values
  * Possible outliers
  * Imbalanced classes

---

# 🛠️ End-to-End Project Pipeline

## 1️⃣ Data Collection & Understanding

### Goal

Understand the dataset structure and target variable.

### Tasks

* Load dataset
* Inspect rows and columns
* Identify feature types
* Understand target labels
* Check dataset size
* Explore class distribution

### Files

```bash
src/data/load_data.py
notebooks/exploration.ipynb
```

---

# 2️⃣ Data Cleaning

### Goal

Fix raw data issues before preprocessing.

### Tasks

* Handle missing values
* Remove duplicates
* Fix inconsistent formatting
* Detect invalid values
* Handle noisy records

### Important Note

Cleaning happens BEFORE preprocessing.

### Files

```bash
src/data/preprocess.py
```

---

# 3️⃣ Exploratory Data Analysis (EDA)

### Goal

Understand patterns and relationships inside the data.

### Tasks

* Univariate analysis
* Correlation analysis
* Distribution visualization
* Outlier analysis
* Relationship between lifestyle habits and mental health

### Tools

* Pandas
* Matplotlib
* Seaborn

### Files

```bash
notebooks/exploration.ipynb
```

---

# 4️⃣ Feature Engineering

### Goal

Create better features for the model.

### Possible Features

* Sleep-to-screen-time ratio
* Activity score
* Lifestyle risk score

### Tasks

* Create derived features
* Encode categorical variables
* Feature selection

### Files

```bash
src/features/feature_engineering.py
```

---

# 5️⃣ Data Preprocessing

### Goal

Prepare clean data for machine learning models.

### Tasks

* Scaling numerical features
* Encoding categorical features
* Train-test split
* Handle class imbalance
* Build preprocessing pipeline

### Tools

* StandardScaler
* OneHotEncoder
* SMOTE

### Files

```bash
src/data/preprocess.py
```

---

# 6️⃣ Model Training

### Goal

Train and compare machine learning models.

### Models

* Logistic Regression
* Random Forest
* XGBoost
* Neural Network (Optional)

### Tasks

* Train models
* Hyperparameter tuning
* Cross-validation

### Files

```bash
src/models/train.py
src/models/model.py
```

---

# 7️⃣ Model Evaluation

### Goal

Evaluate model performance.

### Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Tasks

* Compare models
* Select best model
* Analyze errors

### Files

```bash
src/models/train.py
```

---

# 8️⃣ Model Saving

### Goal

Save the trained model for deployment.

### Tasks

* Export trained model
* Save preprocessing objects

### Files

```bash
saved_models/
```

---

# 9️⃣ Prediction System

### Goal

Make predictions using new user input.

### Tasks

* Load trained model
* Process user input
* Return prediction

### Files

```bash
src/models/predict.py
```

---

# 🔟 GUI Development

### Goal

Build an interactive interface for users.

### Features

* User input form
* Prediction display
* Simple and clean UI

### Technologies

* Streamlit / Tkinter

### Files

```bash
gui/app.py
```

---

# 1️⃣1️⃣ Final Integration

### Goal

Connect all project components together.

### Tasks

* Connect preprocessing with model
* Connect model with GUI
* End-to-end testing

### Files

```bash
main.py
```

---

# 📁 Project Structure

```bash
data/
├── raw/
├── processed/

notebooks/

src/
├── data/
├── features/
├── models/
├── utils/

gui/

saved_models/

main.py
requirements.txt
README.md
```

---

# 🚀 How to Run

## Install Requirements

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python main.py
```

---

# 📈 Future Improvements

* Web deployment
* Deep learning improvements
* Real-time analytics
* Recommendation system
* Explainable AI (XAI)

---

# 👨‍💻 Team

* Youssef George
* Mostafa Mohammed Khedr
* Hady Ayman
* Mohammed Ashraf
