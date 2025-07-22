from . import db

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    vector = db.Column(db.ARRAY(db.Float), nullable=False)
