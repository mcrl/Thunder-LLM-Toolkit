from datasets import Dataset


# openbookqa
def process_obqa(obqa: Dataset) -> Dataset:
    def _process_obqa(doc):
        answerKey = doc['answerKey']
        label_index = doc['choices']['label'].index(answerKey)
        options = doc['choices']['text']
        out_doc = {
            "question": doc['question_stem'],
            "correct": options[label_index],
            "incorrects": [options[i] for i in range(len(options)) if i != label_index],
        }
        return out_doc
    
    obqa_processed = obqa.map(_process_obqa, remove_columns=obqa.column_names)
    return obqa_processed
