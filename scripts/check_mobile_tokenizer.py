"""Compare the Objective-C BPE algorithm with the captured Qwen prompt."""

import json
from pathlib import Path

import numpy as np
import regex
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
tokenizer = AutoTokenizer.from_pretrained(ROOT / "upstream/ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B")
definition = json.loads((ROOT / "ios/Tokenizer/tokenizer.json").read_text(encoding="utf-8"))
pattern = definition["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
vocab = definition["model"]["vocab"]
ranks = {tuple(pair): index for index, pair in enumerate(definition["model"]["merges"])}
base = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
codes = base + [256 + index for index, value in enumerate(range(256)) if value not in base]
base += [value for value in range(256) if value not in base]
symbols = {value: chr(code) for value, code in zip(base, codes)}


def mobile_ids(content: str) -> list[int]:
    ids = []
    for match in regex.finditer(pattern, content):
        tokens = [symbols[value] for value in match.group().encode("utf-8")]
        while len(tokens) > 1:
            choices = [(ranks.get((tokens[index], tokens[index + 1]), 10**9), index)
                       for index in range(len(tokens) - 1)]
            rank, index = min(choices)
            if rank == 10**9:
                break
            tokens[index:index + 2] = [tokens[index] + tokens[index + 1]]
        ids += [vocab[token] for token in tokens]
    return ids


caption = (ROOT / "ios/Tokenizer/sample-caption.txt").read_text(encoding="utf-8").strip()
prompt = ("# Instruction\nFill the audio semantic mask based on the given conditions:"
          "\n\n# Caption\n" + caption + "\n\n# Metas\n- bpm: N/A\n"
          "- timesignature: N/A\n- keyscale: N/A\n- duration: 30 seconds\n")
actual = mobile_ids(prompt)
expected = tokenizer(prompt, add_special_tokens=False).input_ids
print("mobile", len(actual), "reference", len(expected))
for index, (left, right) in enumerate(zip(actual, expected)):
    if left != right:
        print("first difference", index, left, right, repr(tokenizer.decode([left])), repr(tokenizer.decode([right])))
        break
else:
    print("prefix equal", len(actual) == len(expected))
assert np.array_equal(tokenizer(prompt + "<|endoftext|>\n").input_ids,
                      np.fromfile(ROOT / "ios/Probe/encoder-input-ids.i64", dtype="<i8"))
assert actual + [151643, 198, 151643] == tokenizer(prompt + "<|endoftext|>\n").input_ids
