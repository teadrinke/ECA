import json


with open('output_1.jsonl', 'r') as f:
    lines = list(f)

training_data: list = []

for line in lines:
    row = json.loads(line)
    training_data.append(  [ row["text"], { "entities": row["entities"] } ] )

print(len(training_data))

train_split = int(len(training_data) * 0.8) # 80% training and 20% deve set

train_data = training_data[:train_split]
dev_data = training_data[train_split:]

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm


def convert(path, dataset):
    nlp = spacy.blank("en")
    db = DocBin()
    for text, annot in tqdm(dataset): 
        try:
                doc = nlp.make_doc(text) 
                ents = []
                for start, end, label in annot["entities"]:
                    span = doc.char_span(start, end, label=label, alignment_mode="contract")
                    if span is None:
                        print("Skipping nil entity")                
                    if span and span.text != span.text.strip():
                        print("Skipping entity spans with whitespace")
                    elif span:
                        ents.append(span)
                doc.ents = ents

                db.add(doc)
        except Exception as e:
            # print(e)
            continue
    db.to_disk(path)
    
convert("train.spacy", train_data)
convert("dev.spacy", dev_data)