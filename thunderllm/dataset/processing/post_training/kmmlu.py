from datasets import Dataset
import pandas as pd


# kmmlu
def process_kmmlu(kmmlu: Dataset) -> Dataset:
    def _process_kmmlu(doc):
        question = f"{doc['question']}\nA. {doc['A']}\nB. {doc['B']}\nC. {doc['C']}\nD. {doc['D']}\n정답:"
        answer1234 = int(doc["answer"])
        answerABCD = {1: "A", 2: "B", 3: "C", 4: "D"}[answer1234]
        options = {ABCD: ABCD + ": " + doc[ABCD] for ABCD in ["A", "B", "C", "D"]}
        out_doc = {
            "question": question,
            "correct": options[answerABCD],
            "incorrects": [options[i] for i in ["A", "B", "C", "D"] if i != answerABCD],
        }
        return out_doc
    
    columns_to_remove = [col for col in kmmlu.column_names if col not in ["Category", "Human Accuracy"]]
    kmmlu_processed = kmmlu.map(_process_kmmlu, remove_columns=columns_to_remove)
    return kmmlu_processed

def sample_kmmlu_by_category(kmmlu_processed: Dataset, N: int) -> Dataset:
    df = kmmlu_processed.to_pandas()
    sampled_rows = []

    for category, group in df.groupby("Category"):
        if len(group) <= N:
            sampled = group
        else:
            group_sorted = group.sort_values("Human Accuracy", ascending=True)
            half_N = N // 2
            low_acc = group_sorted.iloc[:half_N]
            remaining = group_sorted.iloc[half_N:]
            random_sample = remaining.sample(n=N - half_N, random_state=42)
            sampled = pd.concat([low_acc, random_sample])
        sampled_rows.append(sampled)

    sampled_df = pd.concat(sampled_rows, ignore_index=True)
    kmmlu_sampled = Dataset.from_pandas(sampled_df)
    return kmmlu_sampled
