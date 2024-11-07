from models import db
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os
import utils
import redis
import json
import redis
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import os


load_dotenv()

app = Flask(__name__)
api = Api(app)
CORS(app, origins=['http://localhost:3000','http://localhost:5500/'])

es = Elasticsearch(hosts=[f"http://127.0.0.1:{os.getenv('ES_PORT')}"])

app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
db.init_app(app)


redis_client = redis.StrictRedis(host='localhost', port=6379, db=2)
class ChatBot(Resource):
    def post(self):
        data = request.json
        query = data.get("query")
        response = utils.handleChatBot(query)

        return jsonify({'response': response, 'query': query})
    
class Companies(Resource):
    def get(self):
        companies = utils.get_companies()
        years = sorted(utils.get_years(companies[0]))
        quarters = sorted(utils.get_quarters(companies[0],years[0]))
        speakers = utils.get_speakers(companies[0],years[0],quarters[0])
        designations = utils.get_designation(companies[0],years[0],quarters[0],speakers[0])
        return jsonify({'companies': companies,'company':companies[0],'years':years,'year':years[0],'quarters':quarters,'quarter':quarters[0],'speakers':speakers,'speaker':speakers[0], 'designation':designations[0]})
    

class Years(Resource):
    def get(self):
        company_name = request.args.get('company')
        years = sorted(utils.get_years(company_name))
        quarters = sorted(utils.get_quarters(company_name,years[0]))
        speakers = utils.get_speakers(company_name,years[0],quarters[0])
        designations = utils.get_designation(company_name,years[0],quarters[0],speakers[0])
        return jsonify({'years':years,'year':years[0],'quarters':quarters,'quarter':quarters[0],'speakers':speakers,'speaker':speakers[0],'designation':designations[0]})



class Quarters(Resource):
    def get(self):
        company_name = request.args.get('company')
        selected_year = request.args.get('year')
        quarters = sorted(utils.get_quarters(company_name, selected_year))
        speakers = utils.get_speakers(company_name, selected_year, quarters[0])
        designations = utils.get_designation(company_name,selected_year,quarters[0],speakers[0])
        return jsonify({'quarters': quarters, 'quarter': quarters[0], 'speakers': speakers, 'speaker': speakers[0],'designation':designations[0]})


class Speakers(Resource):
    def get(self):
        company_name = request.args.get('company')
        selected_year = request.args.get('year')
        selected_quarter = request.args.get('quarter')
        speakers = utils.get_speakers(
            company_name, selected_year, selected_quarter)
        designations = utils.get_designation(company_name,selected_year,selected_quarter,speakers[0])
        return jsonify({'speakers': speakers, 'speaker': speakers[0],'designation':designations[0]})
    
class Designation(Resource):
    def get(self):
        company_name = request.args.get('company')
        selected_year = request.args.get('year')
        selected_quarter = request.args.get('quarter')
        speaker = request.args.get('speaker')

        designations = utils.get_designation(company_name,selected_year,selected_quarter,speaker)
        return jsonify({'designation': designations[0]})
        


class Search(Resource):
    def get(self):
        query = request.args.get("q").lower()
        tokens = query.split(" ")

        clauses = [
            {
                "span_multi": {
                    "match": {"fuzzy": {"text": {"value": i, "fuzziness": "AUTO"}}}
                }
            }
            for i in tokens
        ]

        payload = {
            "bool": {
                "must": [{"span_near": {"clauses": clauses, "slop": 0, "in_order": False}}]
            }
        }

        resp = es.search(index="topics", query=payload, size=5000)

        hits = resp['hits']['hits']
        line_chart_data, pie_chart_data = utils.elasticsearch_visualisation(
            hits)
        response = [{'Ticker': result['_source']['ticker'],
                     'Company': result['_source']['company'],
                     'Paragraph': result['_source']['text'],
                     'Year': result['_source']['year'],
                     'Quarter': result['_source']['quarter'],
                     'Speaker': result['_source']['speaker'],
                     'Designation': result['_source']['designation'],
                     } for result in hits]

        response.append({'Figure_1': line_chart_data,
                        'Figure_2': pie_chart_data})
        return jsonify(response)


class Dashboard(Resource):

    def post(self):
        data = request.json
        company = data.get("company")
        year = data.get("year")
        quarter = data.get("quarter") 
        speaker = data.get("speaker") 
        designation = data.get("designation")


        years = sorted(utils.get_years(company))
        if (year is None):
            year = years[0]

        quarters = sorted(utils.get_quarters(company, year))
        if (quarter is None):
            quarter = quarters[0]

        speakers = utils.get_speakers(company, year, quarter)
        if (speaker is None):
            speaker = speakers[0]

        designations = utils.get_designation(company,year,quarter,speaker)
        if (designation is None):
            designation = designations[0]

        cached_data = redis_client.get(f'{company}_{year}_{quarter}_{speaker}')

        if cached_data:
            response = json.loads(cached_data)
            return response


        ner, word_frequencies, topics_data, sentiment_data, sentiment_para = utils.generate_visualizations(
            company, year, quarter, speaker)


        response = {
            
            "company": company,
            "paramOptions": {
                 "quarters": quarters,
                 "speakers": speakers,
                 "years": years
             },

            "selectedParams": {
                    "quarter": quarter,
                    "speaker": speaker,
                    "year": year,
                    "designation":designation
             },

             "visualisations": [
                 {
                     "name": "word_cloud",
                     "data": word_frequencies

                 },
                 {
                     "name": "ner_and_sentiment",
                     "data": {
                         "ner": ner,
                         "sentiment_para": sentiment_para
                     }

                 },
                 {
                     "name": "topics",
                     "data": topics_data
                 },
                 {
                     "name": "sentiment_data",
                     "data": sentiment_data
                 }

             ]
        }

        redis_client.set(f"{company}_{year}_{quarter}_{speaker}",json.dumps(response))
        return jsonify(response)

class Home(Resource):
    def get(self):
        companies = utils.get_companies()
        return jsonify({'companies': companies})

api.add_resource(Companies, '/get_companies')
api.add_resource(Years, '/get_years')
api.add_resource(Quarters, '/get_quarters')
api.add_resource(Speakers, '/get_speakers')
api.add_resource(Designation, '/get_designations')
api.add_resource(Search, '/search')
api.add_resource(Dashboard, '/dashboard')
api.add_resource(ChatBot, '/chatbot')
api.add_resource(Home, '/')

if __name__ == "__main__":
    app.run(debug=True, port=8080)
