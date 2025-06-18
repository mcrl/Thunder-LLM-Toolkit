def doc_to_text(doc):
    return 0


def doc_to_target(doc):
    idx = doc["text"].index("_") + 1
    return doc["text"][idx:].strip()


def doc_to_choice(doc):
    idx = doc["text"].index("_")
    options = [doc["answer"], doc["candidate"]]
    return [doc["text"][:idx] + opt for opt in options]
