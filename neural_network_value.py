import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model
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
    print(X.shape)
    model = Sequential()
    model.add(Dense(60, input_dim=29, kernel_initializer="normal", activation='relu'))
    model.add(Dense(30, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(15, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(8, activation='linear'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=["mse", "mae"])
    print(X_train)
    history = model.fit(X_train, y_train, epochs=50, batch_size=10)
    model.save(f"./models/nn1_{predicting}")

    plt.clf()
    plot_model(model, show_shapes=True, show_layer_names=True)
    plt.savefig("nn_model.png")

    plt.clf()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"./plots/nn_model_loss_{predicting}.png")

    # predicting
    y_predict = model.predict(X_test)
    y_predict = np.array(list(map(lambda x: x[0], y_predict)))

    # plotting
    plt.clf()
    plt.scatter(y_test, y_predict)
    plt.grid()
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title(f"NN Predicting {predicting} ({y.shape[0] - divider} days from past {divider} days of data)")

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

    plt.savefig(f"./plots/nn_predicting_{predicting}.png")


if __name__ == "__main__":
    # main("v(Binance Coin)", 0.01)
    # main("v(Bitcoin)", 0.022)
    # main("v(Ethereum)", 0.01)
    # main("v(Cardano)", 0.01)

    main("v(XRP)", 0.01)

    # main("v(Dogecoin)", 0.001)
    # main("v(Tether)")
    # main("v(USD Coin)")
    # main("v(Polkadot)")
    # main("v(Solana)", 0.01)
