import json
with open("data/space_summ_val.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)
    
with open("data/space_summ_train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

val_train_data = val_data + train_data

with open("data/space_summ_val_train.json", "w", encoding="utf-8") as f:
    json.dump(val_train_data, f, ensure_ascii=False, indent=4)