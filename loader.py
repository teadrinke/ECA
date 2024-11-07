from elasticsearch import Elasticsearch

import csv

es = Elasticsearch(hosts=["http://127.0.0.1:9200"])

print(f"Connected to ElasticSearch cluster `{es.info().body['cluster_name']}`")

with open("Eight_companies.csv", "r") as f:
    reader = csv.reader(f)

    for i, line in enumerate(reader):
        document = {
            "text": line[5],
            "ticker": line[0],
            "company": line[7],
            "year": line[1],
            "quarter": line[2],
            "speaker": line[3],
            "designation": line[4]
        }
        es.index(index="topics", document=document)
es.indices.refresh(index="topics")
print(es.cat.count(index="topics", format="json"))
