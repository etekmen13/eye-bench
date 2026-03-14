import json
from pathlib import Path
import argparse

import pandas as pd
from lupa.lua54 import LuaRuntime


RAW_CSV = Path("data/raw/noita_eye_data_trigrams.csv")
RAW_EYES_DIR = Path("data/raw/eyes")
OUTPUT_JSON = Path("data/processed/eyes.json")


def load_graham_messages(csv_path: Path) -> dict[str, list[int]]:
    df = pd.read_csv(csv_path)

    messages: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        message_id = row["Pos"].replace(" ", "").lower()
        symbols = row.iloc[2:].dropna().astype(int).tolist()
        messages[message_id] = symbols

    return messages


def load_aki_messages(
    eyes_dir: Path,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    lua = LuaRuntime(unpack_returned_tuples=True)

    trigram_messages: dict[str, list[int]] = {}
    unigram_messages: dict[str, list[int]] = {}

    for path in sorted(eyes_dir.iterdir()):
        if not path.is_file():
            continue

        raw = lua.execute(path.read_text())

        if len(raw) % 3 != 0:
            raise ValueError(f"{path.name} does not contain a multiple of 3 symbols.")

        triples = [(raw[i], raw[i + 1], raw[i + 2]) for i in range(1, len(raw) + 1, 3)]

        message_id = path.stem
        unigram_messages[message_id] = [x for triple in triples for x in triple]
        trigram_messages[message_id] = [25 * a + 5 * b + c for a, b, c in triples]

    return trigram_messages, unigram_messages


def validate_messages(
    reference: dict[str, list[int]],
    candidate: dict[str, list[int]],
) -> None:
    if set(reference) != set(candidate):
        missing_from_candidate = set(reference) - set(candidate)
        missing_from_reference = set(candidate) - set(reference)
        raise ValueError(
            "Message IDs do not match.\n"
            f"Missing from candidate: {sorted(missing_from_candidate)}\n"
            f"Missing from reference: {sorted(missing_from_reference)}"
        )

    for message_id in sorted(reference):
        if reference[message_id] != candidate[message_id]:
            raise ValueError(f"Message mismatch for {message_id!r}.")


def build_output(messages: dict[str, list[int]], unigram: bool) -> dict:
    alphabet = sorted({symbol for symbols in messages.values() for symbol in symbols})

    return {
        "alphabet_size": len(alphabet),
        "messages": [
            {
                "message_id": message_id,
                "length": len(symbols),
                "symbols": symbols,
                "unigram": unigram,
            }
            for message_id, symbols in sorted(messages.items())
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--unigram", "-u", action="store_true")

    args = p.parse_args()
    graham_messages = load_graham_messages(RAW_CSV)
    aki_messages, unigram_messages = load_aki_messages(RAW_EYES_DIR)

    validate_messages(graham_messages, aki_messages)
    messages = unigram_messages if args.unigram else graham_messages
    output = build_output(messages, args.unigram)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Messages are identical.")
    print(f"Alphabet size: {output['alphabet_size']}")
    print(f"Total Length: {sum(len(v) for k, v in graham_messages.items())}")
    print(f"Output written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
