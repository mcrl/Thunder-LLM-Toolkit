from datasets import Dataset


# mmlu
def process_mmlu(mmlu: Dataset) -> Dataset:
    def _process_mmlu(doc):
        doc = doc['train']
        answer_idx = doc['answer']
        choices = doc['choices']
        
        question = doc['question'].strip()
        for c, choice in zip(['A', 'B', 'C', 'D'], choices):
            question += f"\n{c}. {choice}"
        question += "\nAnswer:"

        def _choice_formatting(idx: int) -> str:
            choice = ['A', 'B', 'C', 'D'][idx]
            choice += ". "
            choice += choices[idx]
            return choice

        out_doc = {
            "question": question,
            "correct": _choice_formatting(answer_idx),
            "incorrects": [_choice_formatting(i) for i in range(4) if i != answer_idx],
        }
        return out_doc
    
    mmlu_processed = mmlu.map(_process_mmlu, remove_columns=mmlu.column_names)
    return mmlu_processed
