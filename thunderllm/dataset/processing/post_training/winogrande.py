from datasets import Dataset


def doc_to_text(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]

def doc_to_target(doc):
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()

# winogrande
def process_winogrande(winogrande: Dataset) -> Dataset:
    def _process_winogrande(doc):
        def doc_to_question(doc):
            idx = doc["sentence"].index("_")
            return doc["sentence"][:idx].strip() + " "
        
        def custom_doc_to_choice(doc):
            options = [doc["option1"], doc["option2"]]
            ending = doc_to_target(doc)
            return [opt + " " + ending for opt in options]
        
        correct_index = doc_to_text(doc)
        incorrect_index = 1 - correct_index

        out_doc = {
            "question": doc_to_question(doc),
            "correct": custom_doc_to_choice(doc)[correct_index],
            "incorrects": [custom_doc_to_choice(doc)[incorrect_index]],
        }
        return out_doc
    
    winogrande_processed = winogrande.map(_process_winogrande, remove_columns=winogrande.column_names)
    return winogrande_processed
