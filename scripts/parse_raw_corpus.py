import pandas as pd
from lupa.lua54 import LuaRuntime
import os
from collections import Counter
import json

df_graham = pd.read_csv("data/raw/noita_eye_data_trigrams.csv")


messages_graham = {}

cnt = set()
for _, row in df_graham.iterrows():
    name = row["Pos"].replace(" ", "").lower()
    vals = row.iloc[2:].dropna().astype(int).tolist()
    messages_graham[name] = vals
    cnt = cnt.union(set(vals))
print({k: len(v) for k, v in messages_graham.items()})
print(messages_graham["east1"][:10])

lua = LuaRuntime(unpack_returned_tuples=True)
files = os.listdir("data/raw/eyes")
messages_aki = {}
for file in files:
    with open("data/raw/eyes/" + file, "r") as f:
        raw = lua.execute(f.read())
        data = [(raw[i], raw[i + 1], raw[i + 2]) for i in range(1, len(raw) + 1, 3)]
        data = [sum([a * 25, b * 5, c]) for a, b, c in data]
        messages_aki[file.split(".")[0]] = data
        cnt = cnt.union(set(data))

print({k: len(v) for k, v in messages_aki.items()})
print(messages_aki["east1"][:10])

for k in messages_graham.keys():
    if any(x != y for x, y in zip(messages_graham[k], messages_aki[k])):
        raise ValueError("messages do not match")

print("Messages are identical!")
print(f"alphabet size: {len(cnt)}")

output = {
    "alphabet_size": 83,
    "messages": [
        {"message_id": k, "length": len(v), "symbols": v}
        for k, v in messages_graham.items()
    ],
}

with open("data/processed/eyes.json", "w+") as f:
    json.dump(output, f)

print("Output written to data/processed/eyes.json")
