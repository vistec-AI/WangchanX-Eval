import os
from pathlib import Path
import argparse
import re
import string
from collections import Counter
import pathlib
import argparse
import copy
from collections import defaultdict

import pandas as pd
from tqdm.auto import tqdm
from pythainlp.tokenize import word_tokenize
from judge import openai
from config import system_prompt,model_name,get_bool


parser = argparse.ArgumentParser()
parser.add_argument("-f","--file", help="Path CSV file for evaluate")
args = parser.parse_args()


root_dir=pathlib.Path(__file__).parent.resolve()
save_dir = os.path.join(root_dir,"result")

file=args.file
file_name=Path(file).stem

model_df=pd.read_csv(file)
# Check XQuAD
model_df["context"]
model_df["question"]
model_df["references"]
model_df["predictions"]



"""
XQuAD

Official evaluation script for v1.1 of the SQuAD dataset. """
# fork from https://huggingface.co/spaces/evaluate-metric/squad/blob/main/squad.py
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def cut_thai(text): # Add!!!
        return ' '.join(word_tokenize(text,engine="newmm"))  # For Thai

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(cut_thai(s)))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_score(references, predictions):
    f1 = exact_match = total = 0
    for _ground_truths,_prediction in tqdm(list(zip(references, predictions))):
        total += 1
        ground_truths = [_ground_truths]
        prediction = _prediction
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}

xquad_score = compute_score(model_df["references"].tolist(),model_df["predictions"].tolist())
exact_match=xquad_score["exact_match"]
f1=xquad_score["f1"]
xquad_text=f"""XQuAD score
--------------------
exact_match: {exact_match}
F1: {f1}
--------------------
"""
print(xquad_text)


print("Model: "+model_name)
print("System prompt:\n"+system_prompt)


"""
GPT-4 eval
"""
def get_ans(context: str, question: str,reference_answer: str, prediction_answer: str,show:bool=False):
    user_prompt=f'''Passage: {context}
Question: {question}
Reference Answer: "{reference_answer}"
Prediction Answer: "{prediction_answer}"
'''
    if show:
        print(user_prompt)
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        n=1,
        temperature=0.2,
        # top_p=1,
        # # frequency_penalty=0.0,
        # presence_penalty=1,
        max_tokens=1024,
    )
    return response['choices'][0]['message']['content']


def recheck(context: str, question: str,reference_answer: str, prediction_answer: str,show:bool=False):
    ok4=False
    while ok4==False:
        try:
            output=get_ans(context, question,reference_answer, prediction_answer,show=show)
            q1,q2,q3,q4=get_bool(output)
            ok4=True
        except:
            ok4=False
    return output

save_data=defaultdict(list)

for i in tqdm(list(range(len(model_df)))):
    save_data["model"].append(recheck(model_df["context"][i],model_df["question"][i],model_df["references"][i],model_df["predictions"][i]))

raw_data_df=copy.copy(model_df)

raw_data_df["model_gpt4"]=save_data["model"]
_temp = {"q"+str(i+1):[] for i in range(0,4)}
temp_bool=[get_bool(i) for i in save_data["model"]]
for i in tqdm(temp_bool):
    _temp["q1"].append(int(bool(i[0])))
    _temp["q2"].append(int(bool(i[1])))
    _temp["q3"].append(int(bool(i[2])))
    _temp["q4"].append(int(bool(i[3])))
raw_data_df["model_q1"] = _temp["q1"]
raw_data_df["model_q2"] = _temp["q2"]
raw_data_df["model_q3"] = _temp["q3"]
raw_data_df["model_q4"] = _temp["q4"]
save_csv=os.path.join(save_dir,f"eval-{model_name}-{file_name}.csv")
print(f"Save to {save_csv}")
raw_data_df.to_csv(save_csv,index=False)

print()
for i in list(save_data.keys()):
    print(i)
    print("q1: "+str(raw_data_df[f"{i}_q1"].sum()))
    print("q2: "+str(raw_data_df[f"{i}_q2"].sum()))
    print("q3: "+str(raw_data_df[f"{i}_q3"].sum()))
    print("q4: "+str(raw_data_df[f"{i}_q4"].sum()))
    print("---------------")

count_words=defaultdict(list)
print("Number token (avg)")
for j in raw_data_df["predictions"].tolist():
    count_words[i].append(len(word_tokenize(j,keep_whitespace=False)))
print(str(i)+"\t"+str(sum(count_words[i])/len(count_words[i])))

with open(os.path.join(save_dir,f"result-eval-{model_name}-{file_name}.txt"),"w") as f:
    f.write(xquad_text+"\n")
    f.write("Model: "+model_name+"\n")
    f.write("System prompt:\n"+system_prompt+"\n\n")
    for i in list(save_data.keys()):
        f.write(i+"\n")
        f.write("q1: "+str(raw_data_df[f"{i}_q1"].sum())+"\n")
        f.write("q2: "+str(raw_data_df[f"{i}_q2"].sum())+"\n")
        f.write("q3: "+str(raw_data_df[f"{i}_q3"].sum())+"\n")
        f.write("q4: "+str(raw_data_df[f"{i}_q4"].sum())+"\n")
        f.write("---------------"+"\n")
    f.write(f"Save to {save_csv}\n")
    f.write("Number token (avg)\n")
    f.write(str(i)+"\t"+str(sum(count_words[i])/len(count_words[i])))
