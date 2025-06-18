import re
from datasets import Dataset


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: Dataset) -> Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)

def process_hellaswag(hellaswag: Dataset) -> Dataset:
    def _reprocess_hellaswag(doc):
        out_doc = {
            "question": doc["query"],
            "correct": doc["choices"][int(doc["label"])],
            "incorrects": [doc["choices"][i] for i in range(len(doc["choices"])) if i != int(doc["label"])],
        }
        return out_doc
    
    hellaswag_processed = process_docs(hellaswag)
    hellaswag_processed = hellaswag_processed.select_columns(['query', 'choices', 'label']) 
    hellaswag_reprocessed = hellaswag_processed.map(_reprocess_hellaswag, remove_columns=hellaswag_processed.column_names)
    return hellaswag_reprocessed
