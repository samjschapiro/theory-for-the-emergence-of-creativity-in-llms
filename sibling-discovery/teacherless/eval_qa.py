import logging
import argparse
import os
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

sns.set_theme(style="white")


def complete(s):
    if not s.endswith(">"):
        s = s + ">"
    if not s.startswith("<"):
        s = "<" + s
    return s


class Evaluator:
    def __init__(self, dir_, dataset, data_dir):
        self.dir_ = dir_

        with open(os.path.join(data_dir, dataset, "entities_b1_dict.json"), "r", encoding='utf-8') as f:
            self.entities_b1_dict = json.load(f)
        with open(os.path.join(data_dir, dataset, "entities_b2_dict.json"), "r", encoding='utf-8') as f:
            self.entities_b2_dict = json.load(f)
        with open(os.path.join(data_dir, dataset, "train.json"), "r", encoding='utf-8') as f:
            train_sequences = json.load(f)
            # Clean up the train sequences
            self.train_sequences = []
            for i in range(len(train_sequences)):
                self.train_sequences.append(train_sequences[i]["target_text"].split("<q>")[1])

    def eval_file(self, fn='all_items.json'):
        scores_dict = dict()

        for folder_name in tqdm(os.listdir(self.dir_)):
            if not folder_name.startswith("checkpoint"):
                continue
            
            if fn not in os.listdir(os.path.join(self.dir_, folder_name)):
                continue
            
            with open(os.path.join(self.dir_, folder_name, fn)) as f:
                all_items = json.load(f)

            acc = self.eval_items(all_items)

            test_predicted_answers = acc.pop("test_predicted_answers")
            test_predicted_answers_diversity = acc.pop("test_predicted_answers_diversity")
            scores_dict[folder_name] = [(t, round(sum(acc[t])/len(acc[t]), 3)) for t in acc]
            n_samples = len(test_predicted_answers)
            print(n_samples)
            test_predicted_answers = [ans for ans in test_predicted_answers if ans != ""]
            test_predicted_answers_diversity = [ans for ans in test_predicted_answers_diversity if ans != ""]
            scores_dict[folder_name].append(("test_creativity_score", len(set(test_predicted_answers)) / n_samples))
            scores_dict[folder_name].append(("test_diversity_score", len(set(test_predicted_answers_diversity)) / n_samples))

        # sort via checkpoint step. all folder name are in format "checkpoint-<step>-*"
        temp = []
        for folder_name in scores_dict:
            temp.append((folder_name, scores_dict[folder_name]))
        temp.sort(key=lambda var: int(var[0].split("-")[1]))
        
        return temp

    def eval_items(self, all_items):
        acc = dict()   # maps each type of example to the corresponding list of eval results
        for item in all_items:
            if 'type' not in item:
                t = 'test'
            else:
                t = item['type']
            
            if "model_output" in item:
                pred, gold = item["model_output"], item["target_text"]
            else:
                pred, gold = item["model output"], item["target text"]

            #if t == "test":
            #    print(pred)
            #    print(gold)
            #    print()
            if t == "train":
                if "train_memorization_score" not in acc:
                    acc["train_memorization_score"] = []
                acc["train_memorization_score"].append(self.eval_res(pred, gold))
            elif t == "test":
                if "test_seen_score" not in acc:
                    acc["test_seen_score"] = []
                if "test_unseen_score" not in acc:
                    acc["test_unseen_score"] = []
                if "test_predicted_answers" not in acc:
                    acc["test_predicted_answers"] = []
                if "test_predicted_answers_diversity" not in acc:
                    acc["test_predicted_answers_diversity"] = []
                # seen score
                seen_score = self.get_seen_score(pred)
                acc["test_seen_score"].append(seen_score)
                # validity score
                validity_score = self.get_validity_score(pred)
                if seen_score == 1:
                    unseen_score = 0
                else:
                    if validity_score == 1:
                        unseen_score = 1
                    else:
                        unseen_score = 0
                acc["test_unseen_score"].append(unseen_score)
                acc["test_predicted_answers"].append(pred.split("<q>")[1] if unseen_score == 1 else "")  # Store the predicted answer
                acc["test_predicted_answers_diversity"].append(pred.split("<q>")[1] if validity_score == 1 else "")  # Store the predicted answer
            else:
                raise ValueError(f"Unknown type: {t}")
        return acc
    
    def get_seen_score(self, pred):
        pred = pred.split("<q>")[1]
        # print(pred)
        #print(self.train_sequences[0])
        #print()
        return int(pred in self.train_sequences)
    
    def get_validity_score(self, pred):
        assert pred.count("</a>") == 1
        pred = pred.split("<q>")[1]
        pred = pred.split("</a>")[0]
        try:
            b1, b2, a = pred.split("><")
        except:
            print("Failed for parsing:", pred)
            return 0
        b1 = complete(b1)
        b2 = complete(b2)
        a = complete(a)

        try:
            return int((b1 in self.entities_b1_dict[a]) and (b2 in self.entities_b2_dict[a]))
        except:
            print("Failed for dictionary:", a)
            return 0
    
    def eval_res(self, a, b):
        assert b.count("</a>") == 1
        b = b.split("</a>")[0]
        a = a.split("</a>")[0]
        
        return int(a==b)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None, type=str, required=True, help="Input file dir.")
    parser.add_argument("--dataset", default=None, type=str, required=True, help="Dataset name.")
    parser.add_argument("--fn", default='all_items.json', type=str, help="")
    parser.add_argument("--data_dir", default=None, type=str, help="Data dir.")
    args = parser.parse_args()

    evaluator = Evaluator(args.dir, args.dataset, args.data_dir)
    scores_dict = evaluator.eval_file(args.fn)
    temp = []
    for (folder_name, val) in scores_dict:
        temp.append((folder_name, "; ".join(["{}: {}".format(t, res) for (t, res) in val])))

    for (folder_name, res) in temp:
        print(folder_name, "|", res)

    # Plot the results
    data = {
        "Step": [],
        "Score": [],
        "Type": [],
        "Split": []
    }

    for folder_name, scores in scores_dict:
        step = int(folder_name.split('-')[1])
        for score_type, score_value in scores:
            data["Step"].append(step)
            data["Score"].append(score_value)
            data["Type"].append(score_type)
            data["Split"].append("train" if "train" in score_type else "test")

    df = pd.DataFrame(data)
    # Store the dataframe to a csv file
    df.to_csv(os.path.basename(args.dir) + ".csv", index=False)

    # Plotting the results
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # Currently the "train" is dotted in the plot, I want to make it solid
    sns.lineplot(x="Step", y="Score", hue="Type", style="Split", data=df, ax=ax)
    
    ax.set_title("Score by Step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    # Set range of y as 0-1
    ax.set_ylim(0, 1.05)
    # Legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Adjust plot layout
    # Save the plot to a file
    plt.savefig(os.path.basename(args.dir) + ".png")


if __name__ == '__main__':
    main()

