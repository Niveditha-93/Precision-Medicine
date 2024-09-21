"For EDA and Visualization: Libraries"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
"For Preprocessing and Model"
from sklearn.preprocessing import RobustScaler,StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

"Load the Dataset"
df_Breast_cancer = pd.read_csv("data.csv")

"Viewing the Dataset"
Breast_cancer_head = (df_Breast_cancer.head())
Breast_cancer_tail = (df_Breast_cancer.tail())
Breast_cancer_info = (df_Breast_cancer.info())
Breast_cancer_describe = (df_Breast_cancer.describe())

"Checking for Missing values and Outliers"
Breast_cancer_misvalue = df_Breast_cancer.isnull().sum()
"Outliers--valid data point and correct or remove anomalies"
radius_mean = df_Breast_cancer['radius_mean'].quantile(0.25)
perimeter_mean = df_Breast_cancer['perimeter_mean'].quantile(0.50)
area_mean = df_Breast_cancer['area_mean'].quantile(0.75)
'IQR(Inter-Quantile Range) = Middle Range'
IQR = area_mean-radius_mean
lower_bound = radius_mean-1.5*IQR
upper_bound = area_mean+1.5*IQR

'Outlier-visualize'
plt.boxplot(df_Breast_cancer['radius_mean'],vert=False)
plt.title("outlier- Radius_mean")
plt.tight_layout()
plt.show()

plt.boxplot(df_Breast_cancer['area_mean'],vert=False)
plt.title("outlier- area_mean")
plt.tight_layout()
plt.show()

"EDA--Exploratory Data Analysis"
df_Breast_cancer.drop(['id'], axis=1, inplace=True)

'Uni variate- Analysis and Visualize'
df_Breast_cancer_col = df_Breast_cancer.columns
'Sorting Values'
df_Breast_cancer_sotval = df_Breast_cancer['concavity_worst'].sort_values(ascending=True).head(10)
sns.histplot(data=df_Breast_cancer, x='concavity_worst', hue='diagnosis', kde=True)
plt.show()

'Bivariate - Analysis and visualize'
sns.scatterplot(x='concavity_mean', y='concave points_mean', hue='diagnosis', data=df_Breast_cancer)
plt.title('Scatter Plot concavity_mean & concave points_mean')
plt.show()

sns.scatterplot(x='fractal_dimension_se', y='fractal_dimension_worst', hue='diagnosis', data=df_Breast_cancer)
plt.title('Scatter Plot fractal_dimension_se & fractal_dimension_worst')
plt.show()

sns.scatterplot(x='radius_mean', y='radius_worst', hue='diagnosis', data=df_Breast_cancer, palette='Set1')
plt.xlabel('radius_mean')
plt.ylabel('radius_worst')
plt.title('Scatter Plot radius')
plt.show()

'Multivariate Analysis and visualize'

'Correlation Matrix and visualize'
corr_matrix = df_Breast_cancer.corr(numeric_only=True,method="spearman")
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='rainbow',linewidths=.5, linecolor='black')
plt.show()

"Feature Engineering and Visualize"
df_Breast_cancer['arp_mean'] = df_Breast_cancer['area_mean']+df_Breast_cancer['radius_mean']+df_Breast_cancer['perimeter_mean']
df_Breast_cancer['arp_worst'] = df_Breast_cancer['perimeter_worst'] + df_Breast_cancer['area_worst'] + df_Breast_cancer['radius_worst']

sns.histplot(data=df_Breast_cancer, x='arp_mean', hue='diagnosis', kde=True)
plt.show()
sns.histplot(data=df_Breast_cancer, x='arp_worst', hue='diagnosis', kde=True)
plt.show()

df_Breast_cancer['rad_per_se'] = df_Breast_cancer['radius_se'] + df_Breast_cancer['perimeter_se']
sns.scatterplot(data=df_Breast_cancer, x='rad_per_se',y='concavity_mean',hue='diagnosis',palette='tab10')
plt.show()

"Data Preprocessing"

"Log Transformation and visualize"
df_Breast_cancer['LOG_arp_mean'] = np.log1p(df_Breast_cancer['arp_mean'])
df_Breast_cancer['LOG_arp_worst'] = np.log1p(df_Breast_cancer['arp_worst'])
df_Breast_cancer['LOG_rad_per_se'] = np.log1p(df_Breast_cancer['rad_per_se'])


sns.histplot(data=df_Breast_cancer, x='LOG_arp_mean', hue='diagnosis', kde=True)
plt.title("LOG_arp_mean")
plt.show()
sns.histplot(data=df_Breast_cancer, x='LOG_arp_worst', hue='diagnosis', kde=True)
plt.title("LOG_arp_worst")
plt.show()
sns.histplot(data=df_Breast_cancer, x='LOG_rad_per_se', hue='diagnosis', kde=True)
plt.title("LOG_rad_per_se")
plt.show()

'value_counts and visualize'
df_Breast_cancer_val_count = df_Breast_cancer.diagnosis.value_counts(normalize=True)
df_Breast_cancer_val_count['diagnosis'] = df_Breast_cancer['diagnosis'].map({'M': 1, 'B': 0})
mapped = df_Breast_cancer.diagnosis.value_counts()
sns.countplot(data=df_Breast_cancer, x='diagnosis',color='pink')
plt.show()

'RobustScaler'
x = df_Breast_cancer.drop('diagnosis', axis=1)
y = df_Breast_cancer['diagnosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
RS_df_Breast_cancer = RobustScaler(quantile_range=(20, 80))
RS_df_Breast_cancer.fit(x_train)
x_train = RS_df_Breast_cancer.transform(x_train)
x_test = RS_df_Breast_cancer.transform(x_test)

'Label Encoding'
le = LabelEncoder()
df_Breast_cancer['diagnosis'] = le.fit_transform(df_Breast_cancer['diagnosis'])
data = df_Breast_cancer.drop('diagnosis', axis=1)

'StandardScaler'
features = df_Breast_cancer.columns
x = df_Breast_cancer.loc[:, features].values
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_Breast_cancer[['diagnosis']])


"Evaluating the Model"

'Principal Component Analysis and visualize'
pca = PCA(n_components=1)
pca_model = pca.fit_transform(data_scaled)
final_df = pd.DataFrame(data = pca_model, columns=["new column1"])
concatenate = pd.concat([df_Breast_cancer[['diagnosis']],final_df],axis=1)
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio)
cumulative_variance = explained_variance_ratio.cumsum()
print(cumulative_variance)

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

'Random Forest Classifier'
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(final_df, df_Breast_cancer[['diagnosis']], test_size=0.2, random_state=42)
df_Breast_cancer_pca = pca.fit(x_train, y_train)

clf = RandomForestClassifier(random_state=42)
clf.fit(x_train_pca, y_train_pca)
y_pred = clf.predict(x_test_pca)
print(y_pred)

'Metrics'
accuracy = accuracy_score( y_test_pca,y_pred)
print(accuracy)
f1_score = f1_score(y_test_pca,y_pred)
print(f1_score)

'K-Mean Clustering and visualize'
'Euclidean distance calculation'
Euclidean_distance_1 = df_Breast_cancer['arp_mean']
Euclidean_distance_2 = df_Breast_cancer['arp_worst']
distance_cal = np.sqrt(np.sum((np.array(Euclidean_distance_1) - np.array(Euclidean_distance_2)) ** 2))

clustering = df_Breast_cancer[['concave points_mean', 'concave points_se','concave points_worst']]
kmeans = KMeans(n_clusters=5, random_state=42)
Fit_kmeans =kmeans.fit(clustering)
labelling_cluster = df_Breast_cancer['diagnosis'] =kmeans.labels_
plt.scatter(df_Breast_cancer['concave points_mean'], df_Breast_cancer['concave points_se'],df_Breast_cancer['concave points_worst'],c=df_Breast_cancer['diagnosis'], cmap='prism_r')
plt.title('K-means Clustering')
plt.show()

'Metrics'
'Silhouette Score'
silhou_score = silhouette_score(clustering, labelling_cluster)
print(silhou_score)

'Hyperparameter Tunning'
param_grid = [{'penalty':['l1', 'l2'],'C':[.01, 0.1, 1, 5, 10],'solver':['liblinear','saga'],'max_iter': [10000]}]
logistic_regression = LogisticRegression(max_iter=10000)
grid_search = GridSearchCV(estimator=logistic_regression, param_grid=param_grid, cv=5, scoring='accuracy' )
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)









































         
