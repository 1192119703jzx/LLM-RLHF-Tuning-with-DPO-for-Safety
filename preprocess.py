import pandas as pd
import json

df = pd.read_json("train.jsonl", lines=True)

df = df[df["response_0"].str.len() > 10]
df = df[df["response_1"].str.len() > 10]

# only need true/ture responses
df = df[df["is_response_0_safe"] ^ df["is_response_1_safe"]]
print(len(df))
df = df.sample(n=2500, random_state=42)

df = df.dropna(subset=["prompt", "response_0", "response_1", "safer_response_id"])

# Create pairwise preference samples
def create_pair(row):
    if row["safer_response_id"] == 0:
        return {"prompt": row['prompt'], "chosen": row["response_0"], "rejected": row["response_1"]}
    else:
        return {"prompt": row['prompt'], "chosen": row["response_1"], "rejected": row["response_0"]}
    
paired_dataset = df.apply(create_pair, axis=1).tolist()

# save the paired dataset to a json file
with open("paired_dataset_train_TF.json", "w") as f:
    json.dump(paired_dataset, f, indent=4)
