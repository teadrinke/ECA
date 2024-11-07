from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()


class Transcripts(db.Model):
    __tablename__ = 'transcripts'
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String)
    year = db.Column(db.Integer)
    quarter = db.Column(db.Integer)
    speaker = db.Column(db.String)
    designation = db.Column(db.String)
    paragraph = db.Column(db.String)
    processed_response = db.Column(db.String)
    company = db.Column(db.String)
