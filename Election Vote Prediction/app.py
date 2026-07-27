import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Page settings
st.set_page_config(
    page_title="Election Vote Prediction",
    page_icon="🗳️",
    layout="wide"
)

# Load dataset
data = pd.read_csv("election_data.csv")

# Input and target
X = data[["Previous_Votes", "Voter_Turnout"]]
y = data["Votes"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Heading
st.title("🗳️ Election Vote Prediction")
st.write("Machine Learning Project using Multiple Linear Regression")

st.warning(
    "Educational project using synthetic election-style data. "
    "Results are not real election forecasts."
)

# Sidebar
st.sidebar.title("Menu")

page = st.sidebar.radio(
    "Select Option",
    [
        "Home",
        "Dataset",
        "Election Graph",
        "Vote Prediction",
        "Winner Prediction",
        "Model Accuracy"
    ]
)

# ---------------- HOME ----------------

if page == "Home":

    st.header("Election Vote Prediction System")

    st.write("""
    This project predicts election votes using Machine Learning.

    **Input Features:**
    - Previous Election Votes
    - Expected Voter Turnout

    **Algorithm:**
    - Multiple Linear Regression
    """)

    st.subheader("Project Workflow")

    st.write("""
    Dataset → Data Analysis → Model Training →
    Vote Prediction → Winner Prediction → Result Visualization
    """)


# ---------------- DATASET ----------------

elif page == "Dataset":

    st.header("📊 Election Dataset")

    st.dataframe(data, use_container_width=True)

    st.write("Total Records:", len(data))

    st.subheader("Dataset Statistics")

    st.dataframe(
        data.describe(),
        use_container_width=True
    )


# ---------------- ELECTION GRAPH ----------------

elif page == "Election Graph":

    st.header("📈 Historical Election Vote Trend")

    party_a = data[data["Party"] == "Party_A"]
    party_b = data[data["Party"] == "Party_B"]

    fig, ax = plt.subplots()

    ax.plot(
        party_a["Year"],
        party_a["Votes"],
        marker="o",
        label="Party A"
    )

    ax.plot(
        party_b["Year"],
        party_b["Votes"],
        marker="o",
        label="Party B"
    )

    ax.set_xlabel("Election Year")
    ax.set_ylabel("Votes")
    ax.set_title("Election Vote Trend")

    ax.legend()
    ax.grid()

    st.pyplot(fig)


# ---------------- VOTE PREDICTION ----------------

elif page == "Vote Prediction":

    st.header("🔮 Predict Future Votes")

    previous_votes = st.number_input(
        "Previous Election Votes",
        min_value=0,
        value=200000,
        step=1000
    )

    turnout = st.number_input(
        "Expected Voter Turnout (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    if st.button("Predict Votes"):

        new_data = pd.DataFrame({
            "Previous_Votes": [previous_votes],
            "Voter_Turnout": [turnout]
        })

        prediction = model.predict(new_data)[0]

        prediction = max(0, prediction)

        st.success(
            f"Predicted Votes: {int(prediction):,}"
        )


# ---------------- WINNER PREDICTION ----------------

elif page == "Winner Prediction":

    st.header("🏆 2029 Election Winner Prediction")

    col1, col2 = st.columns(2)

    # Party A
    with col1:

        st.subheader("Party A")

        a_votes = st.number_input(
            "Party A Previous Votes",
            min_value=0,
            value=215000,
            step=1000
        )

        a_turnout = st.number_input(
            "Party A Expected Turnout (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

    # Party B
    with col2:

        st.subheader("Party B")

        b_votes = st.number_input(
            "Party B Previous Votes",
            min_value=0,
            value=188000,
            step=1000
        )

        b_turnout = st.number_input(
            "Party B Expected Turnout (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

    if st.button("Predict Winner"):

        party_a_data = pd.DataFrame({
            "Previous_Votes": [a_votes],
            "Voter_Turnout": [a_turnout]
        })

        party_b_data = pd.DataFrame({
            "Previous_Votes": [b_votes],
            "Voter_Turnout": [b_turnout]
        })

        prediction_a = max(
            0,
            model.predict(party_a_data)[0]
        )

        prediction_b = max(
            0,
            model.predict(party_b_data)[0]
        )

        # Winner
        if prediction_a > prediction_b:
            winner = "PARTY A 🏆"

        elif prediction_b > prediction_a:
            winner = "PARTY B 🏆"

        else:
            winner = "TIE"

        # Total votes
        total_votes = prediction_a + prediction_b

        if total_votes > 0:

            party_a_share = (
                prediction_a / total_votes
            ) * 100

            party_b_share = (
                prediction_b / total_votes
            ) * 100

        else:
            party_a_share = 0
            party_b_share = 0

        # Winning margin
        margin = abs(
            prediction_a - prediction_b
        )

        st.divider()

        st.subheader("Prediction Result")

        result1, result2 = st.columns(2)

        with result1:
            st.metric(
                "Party A Predicted Votes",
                f"{int(prediction_a):,}"
            )

            st.metric(
                "Party A Vote Share",
                f"{party_a_share:.2f}%"
            )

        with result2:
            st.metric(
                "Party B Predicted Votes",
                f"{int(prediction_b):,}"
            )

            st.metric(
                "Party B Vote Share",
                f"{party_b_share:.2f}%"
            )

        st.success(
            f"Predicted Winner: {winner}"
        )

        st.info(
            f"Winning Margin: {int(margin):,} votes"
        )

        # Bar graph
        st.subheader("Predicted Vote Comparison")

        fig, ax = plt.subplots()

        parties = ["Party A", "Party B"]

        predicted_votes = [
            prediction_a,
            prediction_b
        ]

        ax.bar(
            parties,
            predicted_votes
        )

        ax.set_xlabel("Political Party")
        ax.set_ylabel("Predicted Votes")
        ax.set_title(
            "2029 Election Vote Prediction"
        )

        st.pyplot(fig)


# ---------------- MODEL ACCURACY ----------------

elif page == "Model Accuracy":

    st.header("🎯 Model Evaluation")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    test_model = LinearRegression()

    test_model.fit(
        X_train,
        y_train
    )

    predictions = test_model.predict(
        X_test
    )

    score = r2_score(
        y_test,
        predictions
    )

    st.metric(
        "R² Score",
        f"{score:.3f}"
    )

    result = pd.DataFrame({
        "Actual Votes": y_test.values,
        "Predicted Votes": predictions.astype(int)
    })

    st.subheader(
        "Actual vs Predicted Votes"
    )

    st.dataframe(
        result,
        use_container_width=True
    )

    st.caption(
        "The dataset is very small and synthetic, so this score "
        "should not be interpreted as real-world election accuracy."
    )

    #streamlit run app.py