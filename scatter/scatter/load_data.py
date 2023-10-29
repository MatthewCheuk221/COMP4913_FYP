import pandas as pd
import pickle

# data = [[], [], []]
# for i in range(1, 8):
#     df = pd.read_excel(f"Prediction{i}.xlsx")
#     RSSI, actual, predicted = df.iloc[:, :-6].values.tolist(), df.iloc[:, -6:-3].values.tolist(), df.iloc[:, -3:].values.tolist()
#     data[0].extend(RSSI)
#     data[1].extend(actual)
#     data[2].extend(predicted)
#
# df = pd.DataFrame(dict(zip(["raw", "actual", "predicted"], data)))
# file = open("combined.pkl", 'wb')
# pickle.dump(df, file)

with open("combined.pkl", "rb") as file:
    df = pickle.load(file)


# raw = df.iloc[7675, 0]
# print(raw)
# children = []
# for i in range(len(raw) // 32):
#     rssi = raw[i * 32: i * 32 + 16]
#     phase = raw[i * 32 + 16: i * 32 + 32]
#     children.append(f"{i + 1}. RSSI: {rssi}")
#     children.append(f"{i + 1}. Phase: {phase}")

print(df)
