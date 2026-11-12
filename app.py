"""
Titanic Survival Prediction App
Run with: streamlit run app.py

Keep Titanic-Dataset.csv in the same folder as this file,
or upload the CSV through the app.
"""

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
]

# These mappings match LabelEncoder's alphabetical order
# for the standard Titanic dataset.
SEX_MAP = {"female": 0, "male": 1}
EMBARKED_MAP = {"C": 0, "Q": 1, "S": 2}


def prepare_data(df):
    """Clean the Titanic data and create the same features as the notebook."""
    required = [
        "Survived", "Pclass", "Sex", "Age",
        "SibSp", "Parch", "Fare", "Embarked"
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "The CSV is missing these required columns: "
            + ", ".join(missing)
        )

    data = df[required].copy()

    # Fill missing values using the median/mode.
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["Embarked"] = data["Embarked"].fillna(
        data["Embarked"].mode()[0]
    )

    # Convert text categories into numbers.
    data["Sex"] = data["Sex"].astype(str).str.lower().map(SEX_MAP)
    data["Embarked"] = (
        data["Embarked"].astype(str).str.upper().map(EMBARKED_MAP)
    )

    # Remove rows with any remaining unusable values.
    data = data.dropna(subset=required)

    # Create the two engineered features used in the improved notebook.
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    X = data[FEATURES].astype(float)
    y = data["Survived"].astype(int)

    return X, y


@st.cache_resource
def train_model(csv_bytes):
    """Tune a simple Decision Tree and train the final model."""
    df = pd.read_csv(BytesIO(csv_bytes))
    X, y = prepare_data(df)

    # Keep a separate validation set to choose the tree settings.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    depth_options = [3, 4, 5, None]
    leaf_options = [1, 2, 4]

    best_score = -1
    best_params = {}

    for depth in depth_options:
        for leaf in leaf_options:
            candidate = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=42,
            )
            candidate.fit(X_train, y_train)
            validation_predictions = candidate.predict(X_val)
            score = accuracy_score(y_val, validation_predictions)

            if score > best_score:
                best_score = score
                best_params = {
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                }

    # Refit the selected model using all labelled rows for the app.
    final_model = DecisionTreeClassifier(
        **best_params,
        random_state=42,
    )
    final_model.fit(X, y)

    return final_model, best_score, best_params, len(X)


st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

st.title("🚢 Titanic Survival Prediction")
st.write(
    "Enter a passenger's details to predict whether they would have "
    "survived the Titanic disaster."
)

st.info(
    "To use the app, upload the Titanic training CSV, or place "
    "'Titanic-Dataset.csv' in the same folder as app.py."
)

uploaded_file = st.file_uploader(
    "Upload Titanic-Dataset.csv",
    type=["csv"],
)

default_csv = Path(__file__).resolve().parent / "Titanic-Dataset.csv"

csv_bytes = None
if uploaded_file is not None:
    csv_bytes = uploaded_file.getvalue()
elif default_csv.exists():
    csv_bytes = default_csv.read_bytes()

if csv_bytes is None:
    st.warning(
        "Please upload Titanic-Dataset.csv to train the prediction model."
    )
    st.stop()

try:
    with st.spinner("Preparing the data and training the model..."):
        model, validation_accuracy, best_params, training_rows = train_model(
            csv_bytes
        )
except Exception as error:
    st.error(f"Could not train the model: {error}")
    st.stop()

st.success("Model is ready!")
col1, col2 = st.columns(2)
col1.metric("Validation accuracy", f"{validation_accuracy:.1%}")
col2.metric("Training passengers", f"{training_rows}")

st.caption(f"Selected Decision Tree settings: {best_params}")

st.subheader("Passenger Details")

with st.form("passenger_details"):
    passenger_class = st.selectbox(
        "Passenger class",
        options=[1, 2, 3],
        format_func=lambda value: {
            1: "1st Class",
            2: "2nd Class",
            3: "3rd Class",
        }[value],
    )

    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0,
    )
    siblings_spouses = st.number_input(
        "Number of siblings/spouses aboard (SibSp)",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
    )
    parents_children = st.number_input(
        "Number of parents/children aboard (Parch)",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
    )
    fare = st.number_input(
        "Ticket fare",
        min_value=0.0,
        max_value=600.0,
        value=32.0,
        step=1.0,
    )
    embarked = st.selectbox(
        "Port of embarkation",
        options=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"],
    )

    submitted = st.form_submit_button("Predict Survival")

if submitted:
    embarked_code = {
        "Cherbourg (C)": 0,
        "Queenstown (Q)": 1,
        "Southampton (S)": 2,
    }[embarked]

    family_size = siblings_spouses + parents_children + 1
    is_alone = 1 if family_size == 1 else 0

    passenger = pd.DataFrame(
        [[
            passenger_class,
            SEX_MAP[sex.lower()],
            age,
            siblings_spouses,
            parents_children,
            fare,
            embarked_code,
            family_size,
            is_alone,
        ]],
        columns=FEATURES,
    )

    prediction = int(model.predict(passenger)[0])
    probabilities = model.predict_proba(passenger)[0]
    survival_probability = probabilities[
        list(model.classes_).index(1)
    ]

    st.subheader("Prediction")
    if prediction == 1:
        st.success("The model predicts: This passenger would have survived.")
    else:
        st.error("The model predicts: This passenger would not have survived.")

    st.write(
        f"Estimated survival probability: **{survival_probability:.1%}**"
    )
    st.progress(float(survival_probability))

    st.caption(
        "This is an educational machine-learning prediction, not a "
        "historical certainty. The probability is based on the trained model."
    )
