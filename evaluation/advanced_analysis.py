import json

COMP_PATH = "test_results/comparison.jsonl"
with open(COMP_PATH, 'r', encoding='utf-8') as f:
        COMPARASION = json.load(f)

TT_list = []
TF_list = []
base_list = []
for i, item in enumerate(COMPARASION):
    if item["base_category"] == True:
        base_list.append(i)
    if item["TT_category"] == True:
        TT_list.append(i)
    if item["TF_category"] == True:
        TF_list.append(i)


TT_lits = set(TT_list)
TF_list = set(TF_list)
base_list = set(base_list)

# Find the intersection of the two sets
intersection_1 = TT_lits.intersection(base_list)
intersection_2 = TF_list.intersection(base_list)
intersection_3 = intersection_2.intersection(TF_list)

with open("test_results/analysis_result.txt", "w") as f:
    f.write(f"Base harmful: {base_list} \n")
    f.write(f"TT harmful: {TT_list} + \n")
    f.write(f"TF harmful: {TF_list} + \n")
    f.write(f"Intersection of TT and base: {intersection_1} \n")
    f.write(f"Intersection of TF and base: {intersection_2} \n")
    f.write(f"Intersection of TF and TT and base: {intersection_3} \n")
    f.write(f"TT list: {len(TT_list)} \n")
    f.write(f"TF list: {len(TF_list)} \n")
    f.write(f"Base list: {len(base_list)} \n")


TF_PATH = "test_results/test_results_moderation_TF.jsonl"
TT_PATH = "test_results/test_results_moderation_TT_2.jsonl"

with open(TF_PATH, 'r', encoding='utf-8') as f:
    TF_dataset = json.load(f)
with open(TT_PATH, 'r', encoding='utf-8') as f2:
    TT_dataset = json.load(f2)

base_safe_TT = []
base_safe_TF = []
base_safe = []
lower_TF = []
lower_TT = []
lower_base_TT = []
lower_base_TF = []
for i, item in enumerate(TF_dataset):
    if i not in base_list:
        base_safe.append(item["Without_DPO_moderation_score"])
        base_safe_TF.append(item["With_DPO_moderation_score"])
        base_safe_TT.append(TT_dataset[i]["Without_DPO_moderation_score"])
    if item["Improve"][0] == False:
        lower_TF.append(item["With_DPO_moderation_score"])
        lower_base_TF.append(item["Without_DPO_moderation_score"])

for item in TT_dataset:
    if item["Improve"] == False:
        lower_TT.append(item["With_DPO_moderation_score"])
        lower_base_TT.append(item["Without_DPO_moderation_score"])

mean_base_safe = sum(base_safe) / len(base_safe) # 0.0506259271001065
mean_base_safe_TF = sum(base_safe_TF) / len(base_safe_TF) # 0.04195581314637866
mean_base_safe_TT = sum(base_safe_TT) / len(base_safe_TT) # 0.050818018403828734
mean_lower_TF = sum(lower_TF) / len(lower_TF) # 0.08128552969743416
mean_lower_TT = sum(lower_TT) / len(lower_TT) # 0.11561247067539664
mean_lower_base_TF = sum(lower_base_TF) / len(lower_base_TF) # 0.04900733550383195
mean_lower_base_TT = sum(lower_base_TT) / len(lower_base_TT) # 0.08807937432602307
print(mean_base_safe, mean_base_safe_TF, mean_base_safe_TT)
print(mean_lower_TF, mean_lower_TT, mean_lower_base_TF, mean_lower_base_TT)
with open("test_results/analysis_result.txt", "a") as f:
    f.write(f"When the base model return safe output, safe/safe model is {mean_base_safe_TT - mean_base_safe} averagely higher\n")
    f.write(f"When the base model return safe output, safe/unsafe model is {mean_base_safe_TF - mean_base_safe} averagely higher\n")
    f.write(f"When the base model is safer, the safe/safe model is {mean_lower_base_TT - mean_lower_TT} averagely lower\n")
    f.write(f"When the base model is safer, the safe/unsafe model is {mean_lower_base_TF - mean_lower_TF} averagely lower\n")