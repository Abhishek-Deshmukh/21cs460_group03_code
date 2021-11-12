from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def import_data_gtrends(show=False):

    terms = [
        "crypto",
        "cryptocurrency",
        "bitcoin",
        "BTC",
        "ethereum",
        "ETH",
        "binance",
        "BNB",
        "cardano",
        "ADA",
        "tether",
        "USDT",
        "XRP",
        "solana",
        "SOL",
        "polkadot",
        "DOT",
        "USDC",
        "doge",
        "dogecoin",
    ]

    def get_df(name):
        return pd.DataFrame(pd.read_csv(name + ".csv"))

    print("Importing google trends data")
    df = get_df(terms[0])
    for term in tqdm(terms[1:]):
        df = df.merge(get_df(term), "right", on="Week")

    print("Cleaning and filling in values")
    # date stuff
    df["Date"] = pd.to_datetime(df["Week"])
    del df["Week"]

    df.replace("<1", 0.5, inplace=True)

    # removing ": (Worldwide)" from the headings
    renamer = {}
    for term in terms:
        renamer[term + ": (Worldwide)"] = term
    df.rename(columns=renamer, inplace=True)

    # setting to float
    for term in terms:
        df[term] = df[term].astype("float")

    if show:
        # running comparisons between paired terms to remove unrelated terms (decreasing noise)
        pairs = [
            ["crypto", "cryptocurrency"],
            ["bitcoin", "BTC"],
            ["ethereum", "ETH"],
            ["binance", "BNB"],
            ["cardano", "ADA"],
            ["tether", "USDT"],
            ["solana", "SOL"],
            ["polkadot", "DOT"],
            ["doge", "dogecoin"],
        ]
        for pair in pairs:
            plt.clf()
            plt.scatter(df[pair[0]], df[pair[1]])
            plt.title(f"GTrends popularity of {pair[0]} vs {pair[1]}")
            plt.xlabel(pair[0])
            plt.ylabel(pair[1])
            plt.grid()
            plt.savefig(f"./plots/{pair[0]}_{pair[1]}.png")

    # from the above results removing "uncorrelated" terms
    # to_remove = ["BNB", "BTC", "ADA", "DOT", "SOL"]

    # interpolating values for the dates in the middle
    start_date = min(df["Date"])
    end_date = max(df["Date"])
    last_row = []
    incrementor = []
    next_row = []
    for single_date in tqdm(
        [
            d
            for d in (
                start_date + timedelta(n)
                for n in range((end_date - start_date).days + 1)
            )
            if d <= end_date
        ]
    ):
        if (df.Date == single_date).any():
            last_row = np.array(df[df.Date == single_date].values.tolist()[0][:-1])
            try:
                next_row = np.array(
                    df[df.Date == single_date + timedelta(7)].values.tolist()[0][:-1]
                )
                incrementor = (next_row - last_row) / 7
            except:
                break
        else:
            last_row += incrementor
            df.loc[-1] = [*last_row, single_date]
            df.index = df.index + 1
    df.set_index("Date", inplace=True)
    df = df.sort_index()
    if show:
        print(df)
    return df


def import_data_crypto_value(show=False):
    print("Importing crypto values")
    df = pd.DataFrame(pd.read_csv("1.csv"))
    df = df.append(pd.read_csv("2.csv"), ignore_index=True)
    df = df.append(pd.read_csv("3.csv"), ignore_index=True)
    df = df.append(pd.read_csv("4.csv"), ignore_index=True)
    df.fillna(0, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")

    # linear interpolation for the middle days
    print("Filling in the missing values")
    start_date = min(df["Date"])
    end_date = max(df["Date"])
    last_row = []
    incrementor = []
    next_row = []
    for single_date in tqdm(
        [
            d
            for d in (
                start_date + timedelta(n)
                for n in range((end_date - start_date).days + 1)
            )
            if d <= end_date
        ]
    ):
        if (df.Date == single_date).any():
            last_row = np.array(df[df.Date == single_date].values.tolist()[0][1:])
            for i in range(1, 7):
                try:
                    next_row = np.array(
                        df[df.Date == single_date + timedelta(i)].values.tolist()[0][1:]
                    )
                    incrementor = (next_row - last_row) / i
                    break
                except:
                    continue
        else:
            last_row += incrementor
            df.loc[-1] = [single_date, *last_row]
            df.index = df.index + 1

    df.set_index("Date", inplace=True)
    df = df.sort_index()

    if show:
        print(df)
        for coin in df.columns:
            plt.clf()
            plt.plot(df.index, df[coin])
            plt.xlabel("Date")
            plt.ylabel("Value in USD")
            plt.xticks(rotation=25)
            plt.title(f"{coin} value")
            plt.grid()
            plt.savefig(f"./plots/{coin}.png")

    # renaming columns
    renamer = {}
    for coin in df.columns[0:]:
        renamer[coin] = f"v({coin})"
    df.rename(columns=renamer, inplace=True)

    return df


if __name__ == "__main__":
    import_data_gtrends(show=True)
    import_data_crypto_value(show=True)
