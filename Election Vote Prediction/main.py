import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Dataset load
data = pd.read_csv("election_data.csv")

# Input and Output
X = data[["Previous_Votes", "Voter_Turnout"]]
y = data["Votes"]

# Linear Regression Model
model = LinearRegression()

# Train Model
model.fit(X, y)


# 1. Show Dataset
def show_dataset():

    print("\n===== ELECTION DATASET =====")
    print(data)


# 2. Show Election Graph
def show_graph():

    party_a = data[data["Party"] == "Party_A"]
    party_b = data[data["Party"] == "Party_B"]

    plt.plot(
        party_a["Year"],
        party_a["Votes"],
        marker="o",
        label="Party A"
    )

    plt.plot(
        party_b["Year"],
        party_b["Votes"],
        marker="o",
        label="Party B"
    )

    plt.xlabel("Election Year")
    plt.ylabel("Votes")
    plt.title("Election Vote Trend")

    plt.legend()
    plt.grid()

    plt.show()


# 3. Predict Votes
def predict_votes():

    print("\n===== VOTE PREDICTION =====")

    previous_votes = int(
        input("Enter Previous Election Votes: ")
    )

    turnout = float(
        input("Enter Expected Voter Turnout (%): ")
    )

    new_data = pd.DataFrame({
        "Previous_Votes": [previous_votes],
        "Voter_Turnout": [turnout]
    })

    prediction = model.predict(new_data)[0]

    print("\nPredicted Votes:", int(prediction))


# 4. Predict Winner
def predict_winner():

    print("\n===== PARTY A =====")

    a_votes = int(
        input("Enter Party A Previous Votes: ")
    )

    a_turnout = float(
        input("Enter Expected Voter Turnout (%): ")
    )

    print("\n===== PARTY B =====")

    b_votes = int(
        input("Enter Party B Previous Votes: ")
    )

    b_turnout = float(
        input("Enter Expected Voter Turnout (%): ")
    )

    # Party A Data
    party_a_data = pd.DataFrame({
        "Previous_Votes": [a_votes],
        "Voter_Turnout": [a_turnout]
    })

    # Party B Data
    party_b_data = pd.DataFrame({
        "Previous_Votes": [b_votes],
        "Voter_Turnout": [b_turnout]
    })

    # Predictions
    prediction_a = model.predict(party_a_data)[0]
    prediction_b = model.predict(party_b_data)[0]

    print("\n===== PREDICTION RESULT =====")

    print(
        "Party A Predicted Votes:",
        int(prediction_a)
    )

    print(
        "Party B Predicted Votes:",
        int(prediction_b)
    )

    # Winner Prediction
    if prediction_a > prediction_b:

        print("\nPredicted Winner: PARTY A")

    elif prediction_b > prediction_a:

        print("\nPredicted Winner: PARTY B")

    else:

        print("\nPrediction: TIE")


    # Total Predicted Votes
    total_votes = prediction_a + prediction_b

    # Vote Share Percentage
    party_a_percentage = (
        prediction_a / total_votes
    ) * 100

    party_b_percentage = (
        prediction_b / total_votes
    ) * 100

    # Winning Margin
    vote_difference = abs(
        prediction_a - prediction_b
    )

    print("\n===== VOTE ANALYSIS =====")

    print(
        "Party A Vote Share:",
        round(party_a_percentage, 2),
        "%"
    )

    print(
        "Party B Vote Share:",
        round(party_b_percentage, 2),
        "%"
    )

    print(
        "Winning Margin:",
        int(vote_difference),
        "votes"
    )


    # Bar Graph
    parties = ["Party A", "Party B"]

    votes = [
        prediction_a,
        prediction_b
    ]

    plt.bar(parties, votes)

    plt.xlabel("Political Party")
    plt.ylabel("Predicted Votes")

    plt.title(
        "2029 Election Vote Prediction"
    )

    plt.show()


# 5. Model Accuracy
def model_accuracy():

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    test_model = LinearRegression()

    # Train Model
    test_model.fit(
        X_train,
        y_train
    )

    # Prediction
    prediction = test_model.predict(
        X_test
    )

    # R2 Score
    score = r2_score(
        y_test,
        prediction
    )

    print("\n===== MODEL EVALUATION =====")

    print("\nActual Votes:")
    print(y_test.values)

    print("\nPredicted Votes:")
    print(
        prediction.astype(int)
    )

    print(
        "\nR2 Score:",
        round(score, 3)
    )


# Main Menu
while True:

    print("\n================================")
    print("    ELECTION VOTE PREDICTION")
    print("================================")

    print("1. Show Dataset")
    print("2. Show Election Graph")
    print("3. Predict Votes")
    print("4. Predict Winner")
    print("5. Model Accuracy")
    print("6. Exit")

    choice = input(
        "\nEnter Your Choice (1-6): "
    )

    if choice == "1":

        show_dataset()

    elif choice == "2":

        show_graph()

    elif choice == "3":

        predict_votes()

    elif choice == "4":

        predict_winner()

    elif choice == "5":

        model_accuracy()

    elif choice == "6":

        print("\nThank You!")
        print("Program Closed.")
        break

    else:

        print("\nInvalid Choice!")
        print(
            "Please enter a number from 1 to 6."
        )