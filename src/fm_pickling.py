import pickle

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Load all data into X and y

raw_data_path = os.path.join(os.pardir, os.pardir, "data", "raw", "twitchdata-update.csv")
df_twitch = pd.read_csv(raw_data_path)

df_twitch = pd.read_csv('../../data/clean/CleanData.csv')
X = df_twitch.drop(['Followers gained'], axis=1)
y = df_twitch['Followers gained']

# Note: we are not doing a train-test split, since we already have a "final"
# model chosen based on some previous train-test split. We want the best possible
# model, so we fit with the entire training set.

# Instantiate a pipeline that performs all preprocessing steps
pipe = Pipeline(steps=[
    ("transform_precip", PrecipitationTransformer()),
    ("encode_winter", ColumnTransformer(transformers=[
        ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"),
         ["winter_severity_index"])
    ], remainder="passthrough"
    )),
    ("random_forest", RandomForestRegressor())
])

pipeline2 = Pipeline(steps=[
    ("drop_columns", FunctionTransformer(drop_irrelevant_columns)),
    ("transform_text_columns", ColumnTransformer(transformers=[
        ("ohe", OneHotEncoder(categories="auto", handle_unknown="ignore", sparse=False), ["ProductId"]),
        ("summary-tf-idf", TfidfVectorizer(max_features=1000), "Summary"),
        ("text-tf-idf", TfidfVectorizer(max_features=1000), "Text"),
        ("ss", StandardScaler(), make_column_selector(dtype_include=np.number))
    ], remainder="passthrough"))
])

# Fit the pipeline on the full dataset
pipe.fit(X, y)

# Not needed, but print out the coefficients as a way to demonstrate that the
# model was successfully fitted
print("coefficients")
print(pipe.named_steps["random_forest"].feature_importances_)

# Save the fitted pipeline
with open("model.pkl", 'wb') as f:
    pickle.dump(pipe, f)