from langchain_community.document_loaders.csv_loader import CSVLoader
from dotenv import load_dotenv
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

loader = CSVLoader(file_path="Eight_companies.csv")

data = loader.load()
print(data)
vector_store.add_documents(documents=data)
vector_store.save_local("faiss_eca_1")