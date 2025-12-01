# Fake Job Detection using LSTM

**Project type:** Machine learning + Flask web app  
**Owner:** Sinchana N S

## Overview
This project detects whether a job posting is real or fake using an LSTM-based deep learning model.  
It includes scripts for preprocessing, training and evaluating models (Random Forest, XGBoost, LightGBM, LSTM), and a Flask web app to serve predictions.

## Features
- Text preprocessing and word embeddings
- Model training and comparison (Random Forest, XGBoost, LightGBM, LSTM)
- LSTM model deployment using Flask (upload job description or screenshot)
- Simple UI using 	emplates/ and static/
- Secure design notes (project-related): uses server-side checks and avoids storing sensitive data in repo

## Repository structure
\\\
.
├── app.py / mainpage.py        # Flask app (entrypoint)
├── templates/                  # HTML templates
├── static/                     # CSS, JS, images
├── preprocess.py               # text cleaning & tokenization
├── train_lstm.py               # LSTM training
├── evaluate_models.py          # model comparison scripts
├── models/                     # small helper code or model wrappers
├── data/                       # (ignored) raw datasets (not included)
├── lstm_model.keras            # (large model file — handled via Git LFS or external storage)
├── requirements.txt
└── README.md
\\\

## Setup (local)
1. Clone the repo:
\\\
git clone https://github.com/sinchanans/Fake-job-detection-using-LSTM.git
cd Fake-job-detection-using-LSTM
\\\

2. Create and activate Python virtual environment:
\\\
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# or cmd:
.\venv\Scripts\activate.bat
\\\

3. Install dependencies:
\\\
pip install -r requirements.txt
\\\
(If you used \pipreqs\, verify and add any missing packages manually.)

4. Create config / environment file (if your app needs DB credentials):
\\\
# .env (example)
DB_HOST=localhost
DB_USER=root
DB_PASS=yourpassword
\\\
**Do not commit** \.env\ to the repo.

## Running the app
\\\
# run flask app (example)
python mainpage.py 
# or
gunicorn app:app
\\\
Then open \http://127.0.0.1:5000/\ in the browser.

## Notes about model files & large assets
- Large model files (e.g., \lstm_model.keras\) are **not included** in this repo by default to keep the repository lightweight.
- To include large models, use **Git LFS** or host them in cloud storage and provide download instructions.

## Add your own dataset
Place raw data in the \data/\ folder (ignored by \.gitignore\), and run preprocessing & training scripts.

## License
Add your preferred license here (e.g., MIT).

## Contact
Sinchana N S — sinchanagowda4758@gmail.com 
