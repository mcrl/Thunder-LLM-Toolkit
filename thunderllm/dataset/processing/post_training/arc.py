from datasets import Dataset


def process_arc(arc: Dataset) -> Dataset:
    def _process_arc(doc):
        answerKey = doc['answerKey']
        label_index = doc['choices']['label'].index(answerKey)
        # options = doc['choices']['text']
        options = [
            doc['choices']['label'][i] + ": " + doc['choices']['text'][i] for i in range(len(doc['choices']['label']))
        ]
        question = "Question: " + doc['question'] + "\n"
        for opt in options:
            question += opt + "\n"
        question += "Answer: "
        out_doc = {
            "question": question,
            "correct": options[label_index],
            "incorrects": [options[i] for i in range(len(options)) if i != label_index],
        }
        return out_doc

    arc_processed = arc.map(_process_arc, remove_columns=arc.column_names)
    return arc_processed
