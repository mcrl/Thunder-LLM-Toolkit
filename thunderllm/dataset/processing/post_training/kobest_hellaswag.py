from datasets import Dataset


# kobest-hellaswag
def process_kobest_hellaswag(kobest_hellaswag: Dataset) -> Dataset:
    def _process_kobest_hellaswag(doc):
        choices = [
            doc["ending_1"],
            doc["ending_2"],
            doc["ending_3"],
            doc["ending_4"],
        ]
        out_doc = {
            "question": f"문장: {doc['context']}",
            "correct": choices[int(doc["label"])],
            "incorrects": [choices[i] for i in range(len(choices)) if i != int(doc["label"])],
        }
        return out_doc
    
    kobest_hellaswag_processed = kobest_hellaswag.map(_process_kobest_hellaswag, remove_columns=kobest_hellaswag.column_names)
    return kobest_hellaswag_processed
