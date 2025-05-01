from openai import OpenAI
import os
import json
import tqdm

os.environ['OPENAI_API_KEY'] = 'sk-...'

client = OpenAI()

total = 0
correct = 0
correct_mean = 0

with open("test_results/test_results_TF.json", "r") as f:
    TF_dataset = json.load(f)

with open("test_results/test_results_TT_2.json", "r") as f2:
    Tt_dataset = json.load(f2)

def get_flag(text):
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )

    flag = response.results[0].flagged
    return flag


with open("test_results/comparasion.jsonl", "w") as f:
    for itemTF, itemTT in tqdm.tqdm(zip(TF_dataset, Tt_dataset), total=len(TF_dataset)):
        dpo_TF = itemTF["With_DPO"]
        dpo_TT = itemTT["With_DPO"]
        without_dpo = itemTF["Without_DPO"]
        # get flag
        flag_TF = get_flag(dpo_TF)
        flag_TT = get_flag(dpo_TT)
        flag_base = get_flag(without_dpo)
        # add to output
        output = {
            "prompt": itemTF["prompt"],
            "base_category": flag_base,
            "TT_category": flag_TT,
            "TF_category": flag_TF,
        }
        total += 1
        f.write(json.dumps(output, indent=4) + '\n')
        f.flush()
