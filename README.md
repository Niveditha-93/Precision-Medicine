"Breast Cancer Prediction using Machine Learning"

A complete end-to-end machine learning pipeline for breast cancer diagnosis prediction using Exploratory Data Analysis (EDA), Feature Engineering, Dimensionality Reduction, Classification Models, Clustering, and Hyperparameter Optimization.

This project demonstrates how machine learning can assist healthcare professionals in identifying malignant tumors using diagnostic features extracted from medical imaging.

📌 Project Highlights

* End-to-End Data Science Workflow
* Advanced EDA and Data Visualization
* Feature Engineering for better predictive power
* Dimensionality Reduction (PCA)
* Machine Learning Classification (Random Forest)
* Unsupervised Learning (K-Means Clustering)
* Hyperparameter Optimization using GridSearchCV
* Healthcare Data Analytics Application

📊 Project Workflow
Data Collection
      -
Data Exploration (EDA)
      -
Outlier Detection
      -
Feature Engineering
      -
Data Preprocessing
      -
Dimensionality Reduction (PCA)
      -
Machine Learning Model
      -
Model Evaluation
      -
Clustering Analysis
      -
Hyperparameter Optimization
📂 Dataset Information

The dataset contains diagnostic measurements of breast cancer tumors obtained from digitized images of fine needle aspirate (FNA) of breast mass tissue.

Target Variable
Value	Meaning
M	Malignant
B	Benign
Important Features

radius_mean

perimeter_mean

area_mean

concavity_mean

concave points_mean

fractal_dimension

texture_mean

smoothness_mean

These features describe tumor shape, texture, and structure.

🔍 Exploratory Data Analysis (EDA)

EDA helps understand the distribution, patterns, and relationships between tumor features.

Techniques Used

Dataset overview

Statistical summary

Missing value detection

Outlier analysis

Distribution plots

Scatter plots

Correlation heatmap

Example Visualization

Histogram with diagnosis comparison:

sns.histplot(data=df_Breast_cancer, x='concavity_worst', hue='diagnosis', kde=True)
⚠️ Outlier Detection

Outliers were detected using the Interquartile Range (IQR) method.

Formula:

IQR = Q3 - Q1
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR

Visualization performed using Boxplots.

Example:

plt.boxplot(df_Breast_cancer['radius_mean'], vert=False)
🧪 Feature Engineering

New features were created to enhance predictive performance.

Engineered Features
Feature	Formula
arp_mean	area_mean + radius_mean + perimeter_mean
arp_worst	area_worst + radius_worst + perimeter_worst
rad_per_se	radius_se + perimeter_se
Log Transformation

Used to reduce skewness in the distribution.

df_Breast_cancer['LOG_arp_mean'] = np.log1p(df_Breast_cancer['arp_mean'])
⚙️ Data Preprocessing

Several preprocessing techniques were applied.

1️⃣ Label Encoding

Convert categorical diagnosis values into numerical format.

M → 1
B → 0
2️⃣ Robust Scaling

Handles outliers better than standard scaling.

RobustScaler(quantile_range=(20, 80))
3️⃣ Standard Scaling

Used to normalize numerical features.

StandardScaler()
📉 Dimensionality Reduction
Principal Component Analysis (PCA)

PCA reduces feature dimensions while preserving maximum variance.

PCA(n_components=1)
Visualization

Cumulative explained variance plot helps determine optimal number of components.

🤖 Machine Learning Model
Random Forest Classifier

Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines them for improved prediction accuracy.

RandomForestClassifier(random_state=42)

Prediction example:

y_pred = clf.predict(x_test_pca)
📏 Model Evaluation

Model performance was evaluated using:

Accuracy Score

Measures overall prediction correctness.

Accuracy = Correct Predictions / Total Predictions
F1 Score

Balances precision and recall.

accuracy_score()
f1_score()
🔬 Clustering Analysis
K-Means Clustering

Used to identify natural groupings in tumor data.

KMeans(n_clusters=5)
Euclidean Distance

Distance metric used for clustering.

Distance = √Σ(x1 − x2)²

Visualization:

plt.scatter(...)
📊 Clustering Evaluation
Silhouette Score

Measures how well clusters are separated.

Score Range: -1 to 1

Higher values indicate better cluster separation.

silhouette_score()
🔧 Hyperparameter Optimization

GridSearchCV was used to optimize Logistic Regression parameters.

Parameters Tuned
Parameter	Values
penalty	l1, l2
C	0.01, 0.1, 1, 5, 10
solver	liblinear, saga
max_iter	10000

Example:

GridSearchCV(estimator=logistic_regression, param_grid=param_grid)
🧰 Technologies Used
Programming Language

Python

Data Science Libraries
Library	Purpose
Pandas	Data manipulation
NumPy	Numerical computation
Matplotlib	Data visualization
Seaborn	Statistical visualization
Scikit-learn	Machine learning models
📊 Key Visualizations

The project includes:

✔ Histogram distribution plots
✔ Scatter plots
✔ Correlation heatmap
✔ PCA variance plot
✔ Clustering visualization
✔ Boxplots for outlier detection

📁 Project Structure
breast-cancer-ml-analysis
│
├── data.csv
├── breast_cancer_analysis.ipynb
├── breast_cancer_model.py
├── requirements.txt
└── README.md
🚀 Future Improvements

Possible enhancements:

Feature importance analysis

ROC-AUC evaluation

Model comparison (XGBoost, SVM)

Deep learning model implementation

Cross-validation improvements

Deployment using Streamlit or Flask

🎯 Real-World Impact

Breast cancer is one of the most common cancers worldwide.

Machine learning models like this can assist:

Early tumor classification

Clinical decision support

Healthcare data analytics

Predictive diagnostics

👩‍💻 Author

Niveditha B Chandrasekaran

🎓 MSc Biomedical Genetics
📊 Data Scientist | Healthcare Data Analyst | Bioinformatics Enthusiast

🔗 LinkedIn
www.linkedin.com/in/niveditha-b-chandrasekaran

⭐ If you found this project useful, consider starring the repository!
