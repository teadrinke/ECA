from models import Transcripts, db
import spacy
from spacy import displacy
from wordcloud import WordCloud
import pandas as pd
from models import Transcripts
from sqlalchemy import distinct
from spacy.language import Language
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


load_dotenv()


@Language.component("remove_some_ner")
def remove_some_ner(doc):
    original_ents = list(doc.ents)
    for ent in doc.ents:
        if ent.label_ == "WORK_OF_ART" or ent.label_ == "NORP" or ent.label_ == "ORG":
            original_ents.remove(ent)

    doc.ents = original_ents
    return doc


def format_text_with_labels_and_colors(text, entities):
    entities.sort(key=lambda x: x['start'], reverse=True)
    label_colors = {
        'LOCATION': '#FFB6C1',   
        'ORGANIZATION': '#90EE90', 
        'PRODUCT': '#FFD700', 
        'PERSON': '#87CEFA',    
        'PERCENT': '#FF69B4',    
        'DATE': '#00FFFF',    
        'MONEY': '#FFA500',  
        'KPI': '#FFA507'
    }

    for entity in entities:
        start = entity['start']
        end = entity['end']
        label = entity['label']
        entity_text = text[start:end]
        color = label_colors.get(label, '#BED754') 
        formatted_entity = f"<span class='ner-box' style='background-color:{color}; '>{entity_text}({label})</span>"
        text = text[:start] + formatted_entity + text[end:]
    return text


def get_ner(company, year, quarter, speaker):
    nlp = spacy.load("model-best")

    transcripts = Transcripts.query.filter_by(
        company=company, year=year, quarter=quarter, speaker=speaker).all()
    text_data = [transcript.paragraph for transcript in transcripts]
    formatted_text_data = []
    if "remove_some_ner" not in nlp.component_names:
        nlp.add_pipe("remove_some_ner")

    for t in text_data:
        doc = nlp(t)

        json_output = displacy.parse_ents(doc)

        text = json_output['text']
        entities = json_output['ents']
        formatted_text = format_text_with_labels_and_colors(text, entities)

        formatted_text_data.append(formatted_text)

    return formatted_text_data


def get_wordcloud(company, year, quarter, speaker):
    transcripts = Transcripts.query.filter_by(
        company=company, year=year, quarter=quarter, speaker=speaker).all()
    text_data = ' '.join(
        (transcript.processed_response).replace('\n',' ') for transcript in transcripts)

    wordcloud = WordCloud(width=600, height=400)
    wordcloud.generate(text_data)

    word_frequencies = wordcloud.words_

    return word_frequencies


def get_frequency(company, year, quarter, speaker):
    nlp = spacy.load("model-best")
    transcripts = Transcripts.query.filter_by(
        company=company, year=year, quarter=quarter, speaker=speaker).all()
    text_data = ' '.join(
        transcript.processed_response for transcript in transcripts)

    doc = nlp(text_data)


    # LLM = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="llama-3.1-70b-versatile")
    # parser = JsonOutputParser()
    # query = f"""
    # From the list of topics given below, get the top 5 most frequently occuring topics. 
    # Some topics might be worded differently, but should be considered the same.
    # Give the topics along with frequency in JSON format.
    # topics:
    # {text_data}
    # """
    # prompt = PromptTemplate(
    # template="Answer the user query.\n{format_instructions}\n{query}\n",
    # input_variables=["query"],
    # partial_variables={"format_instructions": parser.get_format_instructions()},
    # )

    
    # chain = prompt | LLM | parser

    # result = chain.invoke({"query": query})

    # print(result)


    words = [token.text.lower()
             for token in doc if not token.is_stop and token.is_alpha]
    word_freq = pd.Series(words).value_counts().to_dict()
    topics_data = dict(list(word_freq.items())[:5])

    return topics_data


def get_sentiment_para(model, input_dict):
    outputs = model(**input_dict) 
    probs = torch.nn.functional.softmax(outputs[0], dim=-1)
    return probs


def get_mean_from_proba(proba_list):
    with torch.no_grad():
        stacks = torch.stack(proba_list)
        stacks = stacks.resize(stacks.shape[0], stacks.shape[2])
        mean = stacks.mean(dim=0)
        return mean


def get_proper_proba_list(proba_list):
    with torch.no_grad():
        stacks = torch.stack(proba_list)
        stacks = stacks.resize(stacks.shape[0], stacks.shape[2])
        return stacks


def calc_sentiment(transcript_list):
    proba_list = []
    values_list_rounded = []

    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for para in transcript_list:
        tokens = tokenizer.encode_plus(para, add_special_tokens=False)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        input_dict = {
            'input_ids': torch.Tensor([input_ids]).long().to(device),
            'attention_mask': torch.Tensor([attention_mask]).int().to(device)
        }
        proba = get_sentiment_para(model, input_dict)
        proba_list.append(proba)
    mean = get_mean_from_proba(proba_list)
    values_list_rounded = [round(value, 3)*100 for value in mean.tolist()]

    return values_list_rounded


def calc_sentiment_para(transcript_list):

    proba_list = []
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for para in transcript_list:
        tokens = tokenizer.encode_plus(para, add_special_tokens=False)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        input_dict = {
            'input_ids': torch.Tensor([input_ids]).long().to(device),
            'attention_mask': torch.Tensor([attention_mask]).int().to(device)
        }
        proba = get_sentiment_para(model, input_dict)
        proba_rounded = [[round(val, 3) for val in sublist]
                         for sublist in proba.tolist()]
        proba_list.append(proba_rounded)

    return proba_list


def get_sentiment(company, year, quarter, speaker):
    
    transcript_list = []
    senti_list = []
    transcripts = Transcripts.query.filter_by(
        company=company, year=year, quarter=quarter, speaker=speaker).all()
    transcript_list.extend(
        [transcript.paragraph for transcript in transcripts])

    senti_list = calc_sentiment(transcript_list)
    sentiment_data = {
        "Positive": senti_list[0], "Negative": senti_list[1], "Neutral": senti_list[2]}

    return sentiment_data


def get_sentiment_paragraph(company, year, quarter, speaker):
    
    transcript_list = []
    transcripts = Transcripts.query.filter_by(
        company=company, year=year, quarter=quarter, speaker=speaker).all()
    transcript_list.extend(
        [transcript.paragraph for transcript in transcripts])

    senti_list_para = calc_sentiment_para(transcript_list)
    return senti_list_para


def extract_sentiment_values(senti_para):
    sentiment_values = []
    for para in senti_para:
        positive = para[0][0] * 100  
        negative = para[0][1] * 100  
        neutral = para[0][2] * 100 
        sentiment_values.append({
            "Positive": f"{positive:.2f}%",
            "Negative": f"{negative:.2f}%",
            "Neutral": f"{neutral:.2f}%"})
    return sentiment_values


def generate_visualizations(company, year, quarter, speaker):

    ner = get_ner(company, year, quarter, speaker)

    sentiment_para = get_sentiment_paragraph(company, year, quarter, speaker)
    sentiment_values = extract_sentiment_values(sentiment_para)

    wordcloud_freq = get_wordcloud(company, year, quarter, speaker)

    topics_data = get_frequency(company, year, quarter, speaker)

    sentiment_data = get_sentiment(company, year, quarter, speaker)

    return ner, wordcloud_freq, topics_data, sentiment_data, sentiment_values


def get_designation(company,year,quarter,speaker):
    designations = db.session.query(distinct(Transcripts.designation)).filter_by(
        company=company,year=int(year),quarter=int(quarter),speaker=speaker).all()
    
    designations = [designation[0] for designation in designations]


    return designations



def get_quarters(company_name, selected_year):
    quarters = db.session.query(distinct(Transcripts.quarter)).filter_by(
        company=company_name, year=int(selected_year)).all()

    quarters = [quarter[0] for quarter in quarters]

    return quarters


def get_speakers(company_name, selected_year, selected_quarter):
    speakers = db.session.query(distinct(Transcripts.speaker)).filter_by(
        company=company_name, year=int(selected_year), quarter=int(selected_quarter)).all()
    speakers = [speaker[0] for speaker in speakers]

    return speakers


def get_years(company):
    years = db.session.query(distinct(Transcripts.year)).filter_by(
        company=company).all()

    years = [year[0] for year in years] 

    return years


def get_companies():

    companies = db.session.query(Transcripts.company).distinct()
    companies = [company[0] for company in companies]
    return companies


def elasticsearch_visualisation(hits):
    years_mentions = {}
    company_mentions = {}
    for hit in hits:
        year = hit['_source']['year']
        company = hit['_source']['company']
        years_mentions[year] = years_mentions.get(year, 0) + 1
        company_mentions[company] = company_mentions.get(company, 0)+1

    return years_mentions, company_mentions


def handleChatBot(query):

    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    LLM = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="llama-3.1-70b-versatile", temperature=0.2)
    vector_store = FAISS.load_local(
    "faiss_eca_1", embeddings, allow_dangerous_deserialization=True
    )

    docs = vector_store.similarity_search(query,k=20)

    docs = [doc.page_content for doc in docs]

    prompt = f"""
    You are an Earnings Calls Analyst and you take in user query and answer the query based on the context provided to you.
    If no answer is available in the context, return the answer as "I do not know".
    Do not make up answers if context is not available for the query.

    Query:
    {query}

    Context:
    {docs}
    
    """
    result = LLM.invoke(prompt)
    return result.content
