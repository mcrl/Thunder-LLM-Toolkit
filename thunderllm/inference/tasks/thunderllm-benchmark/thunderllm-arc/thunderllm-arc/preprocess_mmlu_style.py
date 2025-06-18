

def doc_to_choice(doc):
    labels = doc["choices"]["label"]
    options = []
    for i, _ in enumerate(labels):
        options.append(chr(65 + i))
    return options


def doc_to_text(doc):

    question = doc["question"]
    text = "Question: " + question + "\n"
    choice_text = doc["choices"]["text"]
    for i, choice in enumerate(choice_text):
        text += chr(65 + i) + ": " + choice + "\n"
    return text + "Answer: "


def doc_to_target(doc):
    choices = doc["choices"]
    labels = choices["label"]
    letter_to_num = {}
    for i, label in enumerate(labels):
        letter_to_num[label] = i

    answer = letter_to_num[doc["answerKey"]]
    return answer
