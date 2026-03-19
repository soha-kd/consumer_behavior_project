from consumer_behavior_project.preprocessing import load_data, preprocess_data
def main():
    ##print("Consumer behavior project is running!")
    df = load_data("data/online vs store shopping dataset.csv")

    X_train, X_test, y_train, y_test, le = preprocess_data(df)

    print("Preprocessing complete")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

if __name__ == "__main__":
    main()