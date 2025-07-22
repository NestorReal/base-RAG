import os

class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://postgres:@localhost/flask_app_db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = '/tmp'
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    # Add if Gemini uses a separate API key
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

import openai
openai.api_key = "sk-proj-SIu0ItlepIS04KSvNSUCT3BlbkFJo0xxwBNRWrlBIugUa5rQ"
