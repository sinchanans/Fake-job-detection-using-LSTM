import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Store DB in the project root folder (not inside 'instance/')
os.makedirs(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance'), exist_ok=True)

# Correct database path with .db extension
db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance', 'site.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    description = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    flagged = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(150))
    input_text = db.Column(db.Text)  # ✅ use `input_text` to match mainpage.py
    prediction = db.Column(db.String(100))
    image_path = db.Column(db.String(300), nullable=True)
    confidence = db.Column(db.Float)
    Image_path = db.Column(db.String(300), nullable=True)