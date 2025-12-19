import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import json

with open("data/space_summ.json", "r", encoding="utf-8") as f:
    val_test_data = json.load(f)

with open("data/space_summ_splits.txt", "r", encoding="utf-8") as f:
    split_ids = f.read().split("\n")

split_ids = [id.split("\t")[0] for id in split_ids]
val_ids = split_ids[:25]
test_ids = split_ids[25:50]

test_data = []
val_data = []

for i in range(len(val_test_data)):
    sample = val_test_data[i]
    if sample["entity_id"] in val_ids:
        val_data.append(sample)
    else:
        test_data.append(sample)

with open("data/space_summ_test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

with open("data/space_summ_val.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)