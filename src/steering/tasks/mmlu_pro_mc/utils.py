import string
from functools import partial


def doc_to_text(doc):
    text = f"{doc['question']}\n"
    for i in range(len(doc["options"])):
        text += f"{string.ascii_uppercase[i]}. {doc['options'][i]}\n"
    text += "Answer:"
    return text


def doc_to_choice(doc):
    return [string.ascii_uppercase[i] for i in range(len(doc["options"]))]


def process_docs(dataset, subject):
    return dataset.filter(lambda x: x["category"] == subject)


process_math = partial(process_docs, subject="math")
process_law = partial(process_docs, subject="law")
process_history = partial(process_docs, subject="history")
