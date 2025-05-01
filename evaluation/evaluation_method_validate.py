from openai import OpenAI
import os
import json
import tqdm

os.environ['OPENAI_API_KEY'] = 'sk-...'

client = OpenAI()

def moderation_reject(text):
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )

    category_scores = response.results[0].category_scores
    score_dict = category_scores.__dict__
    highest_category = max(score_dict.items(), key=lambda x: x[1])
    mean_score = sum(score_dict.values()) / len(score_dict)

    return highest_category, mean_score

def moderation_chosen(text, category):
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )

    category_scores = response.results[0].category_scores
    score_dict = category_scores.__dict__
    target_score = score_dict[category]
    mean_score = sum(score_dict.values()) / len(score_dict)
    return target_score, mean_score

output = []
total = 0
correct = 0
correct_mean = 0

with open("dataset/paired_dataset_test_TT.json", "r") as f:
    paired_dataset = json.load(f)

with open("dataset/paired_dataset_test_moderation_TT.json", "w") as f:
    for pair in tqdm.tqdm(paired_dataset, total=len(paired_dataset)):
        chosen = pair["chosen"]
        rejected = pair["rejected"]
        is_highest = False
        is_mean = False
        # moderation for rejected
        reject_moderation, reject_mean = moderation_reject(rejected)
        reject_moderation_category = reject_moderation[0]
        reject_moderation_score = reject_moderation[1]
        # moderation for chosen
        chosen_moderation_score, chosen_mean = moderation_chosen(chosen, reject_moderation_category)
        if chosen_moderation_score < reject_moderation_score:
            is_highest = True
            correct += 1
        if chosen_mean < reject_mean:
            correct_mean += 1
            is_mean = True
        # add to output
        output = {
            "prompt": pair["prompt"],
            "category": reject_moderation_category,
            "Rejected_moderation_score": reject_moderation_score,
            "Chosen_moderation_score": chosen_moderation_score,
            "Rejected_mean_score": reject_mean,
            "Chosen_mean_score": chosen_mean,
            "Improve": (is_highest, is_mean)
        }
        total += 1
        f.write(json.dumps(output, indent=4) + '\n')
        f.flush()

print(f"Accuracy: Highest {correct / total * 100:.2f}%; Mean {correct_mean / total * 100:.2f}%")

# Accuracy: Highest 88.73%; Mean 86.99% for safe/unsafe test set
# Accuracy: Highest 60.57%; Mean 56.32% for safe/safe test set
