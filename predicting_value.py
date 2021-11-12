import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
# from data_importer import import_data_gtrends, import_data_crypto_value


# from sklearn.decomposition import KernalPCA


def main(predicting="v(Binance Coin)", test_size=0.01):
    # df1 = import_data_gtrends()
    # df2 = import_data_crypto_value()
    # df = pd.merge(df1, df2, left_index=True, right_index=True)
    # df.to_csv("final_data.csv")
    df = pd.read_csv("final_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    predicting_label = f"predicting_{predicting}"
    prediction_df = pd.DataFrame(
        {"Date": df.index + pd.DateOffset(1), predicting_label: df[predicting]}
    )
    prediction_df.set_index("Date", inplace=True)

    df = pd.merge(df, prediction_df, left_index=True, right_index=True)
    y = df[predicting_label].to_numpy()
    df.drop([predicting_label, predicting], axis=1, inplace=True)
    inputs = df.columns.values
    X = df.to_numpy()
    print(f"{y.shape[0]} days worth of data")

    # splitting data for training and testing
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.4, shuffle=False
    # )

    divider = int(y.shape[0]*(1-test_size))

    X_train = X[:divider]
    X_test = X[divider:]
    y_train = y[:divider]
    y_test = y[divider:]

    # shuffling to remove any affect due to date
    X_train, y_train = shuffle(X_train, y_train)

    # making the model and fitting
    model = LinearRegression().fit(X_train, y_train)

    print("Model score on future test:", model.score(X_test, y_test))
    print("Coefficients")
    for coeff, input_, in zip(model.coef_, inputs):
        print(coeff, input_)

    # predicting
    y_predict = model.predict(X_test)

    # plotting
    plt.clf()
    plt.scatter(y_test, y_predict)
    plt.grid()
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title(f"Predicting {predicting} ({y.shape[0] - divider} days from past {divider} days of data)")

    # x=y line with error around
    max_val = max(max(y_test), max(y_predict))
    min_val = min(min(y_test), min(y_predict))
    prediction_range = np.array([min_val, max_val])
    plt.plot(prediction_range, prediction_range, label="x=y")
    acceptable_error = 0.3
    plt.fill_between(
        prediction_range,
        prediction_range * [1 - acceptable_error],
        prediction_range * [1 + acceptable_error],
        alpha=0.4,
    )

    plt.savefig(f"./plots/predicting_{predicting}.png")


if __name__ == "__main__":
    main("v(Binance Coin)", 0.4)
    main("v(Bitcoin)", 0.022)
    main("v(Cardano)", 0.02)
    main("v(Dogecoin)", 0.01)
    main("v(Ethereum)")
    main("v(Polkadot)")
    main("v(Solana)", 0.1)
    main("v(Tether)")
    main("v(USD Coin)")
    main("v(XRP)", 0.02)
