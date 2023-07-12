# Loan_Approval_Prediction

Certainly! Here's an example of documentation for the code you provided:

# Loan Prediction Code Documentation

This documentation provides an overview of the loan prediction code and explains its functionality. The code uses machine learning algorithms to predict loan approval based on various features.

## Libraries

The following libraries are imported in the code:

- numpy: A library for numerical computing.
- pandas: A library for data manipulation and analysis.
- matplotlib: A library for data visualization.
- seaborn: A library for statistical data visualization.
- sklearn: A library for machine learning.
  - svm: Module for Support Vector Machines algorithm.
  - model_selection: Module for model selection and evaluation.
  - preprocessing: Module for data preprocessing.
  - ensemble: Module for ensemble methods (Random Forest Classifier).
  - linear_model: Module for linear models (Logistic Regression).
- StandardScaler: A class for feature scaling.

## Data Loading

The code loads the loan dataset using the `pd.read_csv()` function and displays the first five rows of the dataset using `loan.head(5)`.

## Data Preprocessing

The following steps are performed to preprocess the data:

- Descriptive Statistics: The `loan.describe()` function is used to compute descriptive statistics, providing insights about the dataset.
- Missing Value Handling: The code checks for missing values using `loan.isnull().sum()` and fills them with appropriate strategies using the `fillna()` method.
- Feature Engineering: Two new features, `LoanAmount_log` and `TotalIncome_log`, are created by applying the natural logarithm transformation on the existing features.
- Categorical Variable Encoding: Categorical variables are encoded using the `LabelEncoder` class from the `sklearn.preprocessing` module.
- Feature Scaling: The features are scaled using the `StandardScaler` class to ensure they are on a similar scale.

## Exploratory Data Analysis (EDA)

The code performs exploratory data analysis using visualizations with `matplotlib` and `seaborn`. The following visualizations are created:

- Count plot of loan applicants grouped by gender.
- Count plot of loan applicants grouped by marital status.
- Count plot of loan applicants grouped by self-employment status.
- Count plot of loan amounts.
- Count plot of credit history.

Insights gained from the visualizations are provided for each plot.

## Model Training and Evaluation

The code splits the data into training and testing sets using the `train_test_split()` function from `sklearn.model_selection`. The machine learning models used in the code are:

- Random Forest Classifier: The model is trained using the `RandomForestClassifier()` class from `sklearn.ensemble`. The accuracy is calculated using `metrics.accuracy_score()`.
- Logistic Regression: The model is trained using the `LogisticRegression()` class from `sklearn.linear_model`. The accuracy is calculated using `metrics.accuracy_score()`.

## Results

The results obtained from the models are as follows:

- Random Forest Classifier Accuracy: 77.84%
- Logistic Regression Accuracy: 80.00%

## Conclusion

In conclusion, the loan prediction code successfully preprocesses the data, performs exploratory data analysis, and trains machine learning models to predict loan approval. The results show that the Logistic Regression model achieved slightly higher accuracy compared to the Random Forest Classifier. However, further analysis and improvements could be made to enhance the model's performance.

## Documentation on GitHub

The documented code along with the explanations and insights can be found in the [Loan Prediction Code Documentation](https://github.com/Adityavenkatramani/Loan_Approval_Prediction).
