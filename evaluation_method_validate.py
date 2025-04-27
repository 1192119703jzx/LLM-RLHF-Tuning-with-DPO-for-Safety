from openai import OpenAI
import os
import json

os.environ['OPENAI_API_KEY'] = "sk-"

client = OpenAI()

def moderation_reject(text):
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )

    category_scores = response.results[0].category_scores
    score_dict = category_scores.__dict__
    highest_category = max(score_dict.items(), key=lambda x: x[1])

    return highest_category

def moderation_chosen(text, category):
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )

    category_scores = response.results[0].category_scores
    score_dict = category_scores.__dict__
    target_score = score_dict[category]
    return target_score

output = []
total = 0
correct = 0

with open("dataset/paired_dataset_test.json", "r") as f:
    paired_dataset = json.load(f)
    #paired_dataset = paired_dataset[:2]
    for pair in paired_dataset:
        chosen = pair["chosen"]
        rejected = pair["rejected"]
        # moderation for rejected
        reject_moderation = moderation_reject(rejected)
        reject_moderation_category = reject_moderation[0]
        reject_moderation_score = reject_moderation[1]
        # moderation for chosen
        chosen_moderation_score = moderation_chosen(chosen, reject_moderation_category)
        if chosen_moderation_score < reject_moderation_score:
            is_corresponding = True
            correct += 1
        else:
            is_corresponding = False
        # add to output
        output.append({
            "prompt": pair["prompt"],
            "category": reject_moderation_category,
            "reject_moderation_score": reject_moderation_score,
            "chosen_moderation_score": chosen_moderation_score,
            "is_corresponding": is_corresponding
        })
        total += 1
    
# save the output to a json file
with open("dataset/paired_dataset_test_moderation.json", "w") as f:
    json.dump(output, f, indent=4)

print(f"Accuracy: {correct / total * 100:.2f}%")

