# from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pprint
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_groq import ChatGroq

load_dotenv()

def normalize(text: str) -> str:
    return " ".join(text.split("\n")).strip()

def convert_to_entity_format(json_data, text, train_data):
    # train_data = []
    entities = {}
    entities = []

    for category, items in json_data.items():

        for item in items:
            start_idx = text.find(item)
            end_idx = start_idx + len(item)
            entities.append((start_idx, end_idx, category))

    train_data.append({"text":text,"entities":entities})

    return train_data

template = """In the sentence below, give me the list of:
- organization named entity
- product named entity
- location named entity
- person named entity
- money named entity
- key performance indicator named entity
- date named entity
- percent named entity

The sentence is in the context of a quarterly earnings call. Carefully read and understand the sentence before extracting entities.

Format the output in JSON with the following keys:
- ORGANIZATION for organization named entity
- PRODUCT for product named entity
- LOCATION for location named entity
- PERSON for person named entity
- MONEY for money named entity
- KPI for key performance indicator named entity
- DATE for date named entity
- PERCENT for percent named entity

Each key should correspond to a list only.
Make sure that each list contains unique items and no two lists can have same item in them.
Only return the output and no additional text.
Sentence below:

{text}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["text"],
)


llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="llama-3.1-70b-versatile", temperature=0)

parser = JsonOutputParser()

chain = prompt | llm | parser

train_data = []
df = pd.read_csv('Eight_companies.csv')

text_list = list(df['Paragraph'])

print(len(text_list))

for text in text_list:
    try:
        output = chain.invoke({"text": normalize(text)})
        print(output)
        train_data = convert_to_entity_format(output, text,train_data)
    except Exception as e:
        print(e)
        break



import json

with open("output_1.jsonl", "a") as f:
    for row in train_data:
       f.write(json.dumps(row) + '\n') 
