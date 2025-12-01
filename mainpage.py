print("✅ Running mainpage.py")

from flask import Flask, render_template, request, redirect, session, flash, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import joblib
import numpy as np
from datetime import datetime
import pytesseract
from PIL import Image
from sqlalchemy import func
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Mail, Message
import re

# --- Keras/TensorFlow Imports for LSTM ---
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- App Initialization and Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

# Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'otpmessagesender@gmail.com'
app.config['MAIL_PASSWORD'] = 'jqzo ffvf ogjd lgmo'
mail = Mail(app)

# Database Configuration
basedir = os.path.abspath(os.path.dirname(__file__))
os.makedirs(os.path.join(basedir, 'instance'), exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'site.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Pytesseract Configuration ---
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception as e:
    print(f"Pytesseract configuration not needed or failed (this is okay on non-Windows): {e}")

# --- Load Final Implemented Model (LSTM) ---
try:
    # Assuming lstm_model.keras and lstm_tokenizer.pkl are in the root project directory
    lstm_model = load_model("lstm_model.keras")
    tokenizer = joblib.load("lstm_tokenizer.pkl")
    MAX_LEN = 300
    print("✅ LSTM model and tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading LSTM model: {e}")
    lstm_model = tokenizer = None
    MAX_LEN = 300

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(150), nullable=False)
    text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(300), nullable=True)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(150))
    category = db.Column(db.String(50))
    message = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    admin_reply = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='New', nullable=False)

# --- Helper Functions for Sending Emails ---
def send_reset_email(user):
    s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    token = s.dumps(user.email, salt='password-reset-salt')
    msg = Message('Password Reset Request',
                  sender=('Fake Job Detector', app.config['MAIL_USERNAME']),
                  recipients=[user.email])
    reset_link = url_for('reset_password', token=token, _external=True)
    msg.body = f'''To reset your password, visit the following link:\n{reset_link}\n\nIf you did not make this request then simply ignore this email and no changes will be made. The link is valid for one hour.'''
    mail.send(msg)

def send_feedback_reply_email(feedback):
    msg = Message('Regarding Your Feedback',
                  sender=('Fake Job Detector Admin', app.config['MAIL_USERNAME']),
                  recipients=[feedback.email])
    msg.body = f'''Hello {feedback.name},\n\nThank you for your feedback. An admin has replied to your message.\n\nYour original message:\n"{feedback.message}"\n\nAdmin's Reply:\n"{feedback.admin_reply}"\n\nWe appreciate your input!\nThanks & Regards \nThe Fake Job Detector Team'''
    mail.send(msg)

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        password_raw = request.form.get("password", "").strip()

        # 1️⃣ Check for empty fields
        if not name or not email or not phone or not password_raw:
            flash("All fields are required!")
            return redirect("/register")

        # 2️⃣ Validate email format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Invalid email format!")
            return redirect("/register")

        # 3️⃣ Validate phone number (10 digits)
        if not re.match(r"^[0-9]{10}$", phone):
            flash("Phone number must be 10 digits!")
            return redirect("/register")

        # 4️⃣ Validate password strength
        if len(password_raw) < 6:
            flash("Password must be at least 6 characters long!")
            return redirect("/register")

        # 5️⃣ Check if user already exists
        if User.query.filter_by(email=email).first():
            flash("User already exists with this email!")
            return redirect("/register")

        # 6️⃣ Save new user
        password = generate_password_hash(password_raw)
        new_user = User(name=name, email=email, phone=phone, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please log in.")
        return redirect("/user_login")

    return render_template("register.html")
@app.route("/user_login", methods=["GET", "POST"])
def user_login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        # 1️⃣ Check for empty fields
        if not email or not password:
            flash("⚠️ Both email and password are required!")
            return redirect("/user_login")

        # 2️⃣ Validate email format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("📧 Invalid email format!")
            return redirect("/user_login")

        # 3️⃣ Check if user exists
        user = User.query.filter_by(email=email).first()
        if not user:
            flash("❌ No user found with this email!")
            return redirect("/user_login")

        # 4️⃣ Verify password
        if not check_password_hash(user.password, password):
            flash("🔐 Incorrect password!")
            return redirect("/user_login")

        # 5️⃣ Successful login
        session["user"] = email
        flash("✅ Login successful! Welcome back.")
        return redirect("/dashboard")

    return render_template("user_login.html")

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = User.query.filter_by(email=email).first()
        if user:
            send_reset_email(user)
            flash('An email has been sent with instructions to reset your password.', 'success')
            return redirect(url_for('user_login'))
        else:
            flash('Email address not found.', 'error')
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    try:
        email = s.loads(token, salt='password-reset-salt', max_age=3600)
    except:
        flash('The password reset link is invalid or has expired.', 'error')
        return redirect(url_for('forgot_password'))
    if request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        if new_password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('reset_password', token=token))
        user = User.query.filter_by(email=email).first()
        user.password = generate_password_hash(new_password)
        db.session.commit()
        flash('Your password has been updated! You can now log in.', 'success')
        return redirect(url_for('user_login'))
    return render_template('reset_password.html')

@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "admin123":
            session["admin"] = "admin"
            return redirect("/admin_dashboard")
        else:
            flash("Invalid admin credentials!")  # Send message to template
            return redirect("/admin_login")       # Reload same page

    return render_template("admin_login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# --- User Dashboard ---
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    user_data = User.query.filter_by(email=session["user"]).first()
    return render_template("dashboard.html", user_data=user_data)

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user" not in session:
        return redirect("/")
    user = User.query.filter_by(email=session["user"]).first()
    message = None
    if request.method == "POST":
        user.name = request.form["name"]
        user.phone = request.form["phone"]
        db.session.commit()
        message = "Profile updated successfully."
    return render_template("my_profile.html", user_data=user, message=message)

@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user' not in session:
        return redirect('/')
    user = User.query.filter_by(email=session['user']).first()
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    if not check_password_hash(user.password, current_password):
        flash('Your current password is not correct.', 'error')
        return redirect('/profile')
    if new_password != confirm_password:
        flash('New password and confirmation do not match.', 'error')
        return redirect('/profile')
    if len(new_password) < 6:
        flash('New password must be at least 6 characters long.', 'error')
        return redirect('/profile')
    user.password = generate_password_hash(new_password)
    db.session.commit()
    flash('Your password has been updated successfully.', 'success')
    return redirect('/profile')

# --- Prediction Routes ---
def get_prediction(text):
    """Helper function to get prediction from the LSTM model and keyword check.

    Returns:
        (label: str, confidence_percent: float, reason: str|None)

    Behaviour:
    - If a high-priority scam phrase is found (and NOT negated nearby),
      return immediate high-confidence Fake.
    - If that scam phrase is negated (e.g. "no registration fee"), fall back to LSTM.
    - Use LSTM probability to produce Fake / Suspicious / Legitimate.
    - Apply post-prediction adjustments:
        * If LSTM predicted Fake but 2+ legit keywords present -> mark Legitimate.
        * If LSTM predicted Suspicious but 2+ legit keywords present -> mark Legitimate.
    """
    import re

    # safety
    if not all([lstm_model, tokenizer]):
        return "Model not available", 0.0, None

    text = (text or "").strip()
    text_lower = text.lower()

    # ---------------- scam phrase list (user-provided + cleaned) ----------------
    scam_phrases = [
        "upfront fee", "processing fee", "security deposit", "training fee",
        "booking charge", "booking fee", "registration charge", "registration fee",
        "registration amount", "registration amount is charged",
        "registration cost", "registration amount is charged",
        "course material charge", "material fee", "investment required",
        "wiring money", "money transfer", "guaranteed job", "daily salary",
        "earn a substantial income", "few minutes of effort", "making extra money",
        "quick money", "easy money", "whatsapp text message only",
        "contact on whatsapp", "telegram interview", "no interview", "pay to join",
        "pay to apply", "service charge", "service fee", "deposit before joining",
        "no resume required", "training charges applicable", "certificate fee",
        "document verification fee", "processing charges apply",
        "background verification fee", "pay for training materials", "registration cost",
        "you can pay", "application fee",

        # WhatsApp/contact-specific patterns
        "whatsapp number", "please send your cv on the same whatsapp number",
        "send cv on whatsapp", "send your resume on whatsapp", "share cv on whatsapp",
        "whatsapp number for interview", "whatsapp the hr", "contact hr on whatsapp",
        "interview on whatsapp", "online interview on whatsapp", "whatsapp only communication",
        "apply through whatsapp", "click interested", "click the interested button",
        "contact me on whatsapp", "address will be shared after sending cv",
        "address will be shared on whatsapp", "send your cv to this whatsapp",
        "share details on whatsapp", "hr maya", "hr will contact you on whatsapp",
        "source: online job portal", "send your cv on whatsapp number", "click interested button"
    ]

    # ---------------- negation patterns (for context-aware checking) ----------------
    NEGATION_PATTERNS = [
        r"no", r"not", r"without", r"does not", r"doesn't",
        r"don't", r"dont", r"never", r"no need", r"no charge",
        r"free of charge", r"not required", r"no fee", r"no fees", r"no charges"
    ]
    # build a fragment for regex (we'll combine words using word boundaries in checks)
    neg_fragment = r"(?:{})".format("|".join([re.escape(p) for p in NEGATION_PATTERNS]))

    def phrase_is_negated(text_check, phrase):
        """Return True if a negation term appears reasonably near the phrase."""
        if not phrase:
            return False
        esc_phrase = re.escape(phrase)
        # check "<negation> ... <phrase>" within ~0-40 characters (covers short phrases)
        pattern = rf"{neg_fragment}[^\n\r]{{0,40}}{esc_phrase}"
        if re.search(pattern, text_check):
            return True
        # also check "<phrase> ... <negation>" (e.g., "registration fee - not required")
        pattern_back = rf"{esc_phrase}[^\n\r]{{0,40}}{neg_fragment}"
        if re.search(pattern_back, text_check):
            return True
        return False

    # ---------------- Step 1: Immediate scam phrase check with negation handling ----------------
    for phrase in scam_phrases:
        if phrase in text_lower:
            if not phrase_is_negated(text_lower, phrase):
                # immediate high-confidence fake
                return "🔴 Fake Job Detected!", 99.9, f"Suspicious keyword found: '{phrase}'"
            else:
                # negated phrase found: log/debug reason and continue to ML
                # (we return reason only at the final output; for debugging we can print)
                # print(f"[debug] Negated scam phrase ignored: '{phrase}'")
                pass

    # ---------------- Step 2: LSTM prediction ----------------
    try:
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=MAX_LEN)
        probability = float(lstm_model.predict(padded)[0][0])
    except Exception as e:
        return "Model prediction error", 0.0, f"Prediction failed: {e}"

    confidence = round(probability * 100, 2)

    if probability > 0.6:
        prediction = "🔴 Fake Job Detected!"
    elif probability > 0.3:
        prediction = "🟡 Suspicious Job"
    else:
        prediction = "🟢 Legitimate Job Posting"

    # ---------------- Step 3: Post-prediction adjustments using legit indicators ----------------
    # Add/extend legitimate keywords as needed for your domain
    legit_keywords = [
        'react', 'python', 'java', 'developer', 'engineer', 'analyst',
        'manager', 'years', 'experience', 'django', 'flask', 'sql', 'excel', 'tableau'
    ]
    legit_keyword_count = sum(1 for keyword in legit_keywords if keyword in text_lower)

    # If model strongly predicted fake but text contains clear legitimate technical indicators
    if prediction == "🔴 Fake Job Detected!":
        if legit_keyword_count >= 2:
            return "🟢 Legitimate Job Posting", confidence, "Corrected from a likely false positive (legit keywords present)."
        elif legit_keyword_count == 1:
            return "🟡 Suspicious Job", confidence, "Model predicted fake, but 1 legit keyword found."

    # If model predicted suspicious, but strong presence of legit keywords -> upgrade to legitimate
    if prediction == "🟡 Suspicious Job" and legit_keyword_count >= 2:
        return "🟢 Legitimate Job Posting", confidence, "Legitimate keywords strongly present."

    # Default return
    return prediction, confidence, None

@app.route("/text_scan", methods=["GET", "POST"])
def text_scan():
    prediction = confidence = text_input = warning = None
    if request.method == "POST":
        text_input = request.form["text_input"]
        prediction, confidence, warning = get_prediction(text_input)
        if "user" in session:
            db.session.add(Scan(user_email=session["user"], text=text_input, prediction=prediction, confidence=confidence))
            db.session.commit()
    return render_template("text_scan.html", prediction=prediction, confidence=confidence, text=text_input, warning=warning)

@app.route("/image_scan", methods=["GET", "POST"])
def image_scan():
    extracted_text = prediction = confidence = warning = None
    if request.method == "POST" and "image_file" in request.files:
        # CORRECTED: Fixed the typo from 'image_.file' to 'image_file'
        image_file = request.files["image_file"]
        if image_file.filename == '':
            return render_template("image_scan.html", error="No file selected.")
        try:
            image = Image.open(image_file)
            extracted_text = pytesseract.image_to_string(image)
            if extracted_text:
                prediction, confidence, warning = get_prediction(extracted_text)
                if "user" in session:
                    db.session.add(Scan(user_email=session["user"], text=extracted_text, prediction=prediction, confidence=confidence, image_path="uploaded"))
                    db.session.commit()
            else:
                return render_template("image_scan.html", error="Could not extract text from the image.")
        except Exception as e:
            print(f"Error processing image: {e}")
            return render_template("image_scan.html", error="Could not process the image file.")
    return render_template("image_scan.html", extracted_text=extracted_text, prediction=prediction, confidence=confidence, warning=warning)

# --- History and Feedback Routes ---
@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/")
    user_data = User.query.filter_by(email=session["user"]).first()
    scans = Scan.query.filter_by(user_email=session["user"]).order_by(Scan.timestamp.desc()).all()
    return render_template("history.html", scans=scans, user_data=user_data)
@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    # Ensure an admin is logged in
    if 'admin' not in session:
        return redirect('/admin_login')

    # Find the user to be deleted
    user_to_delete = User.query.get_or_404(user_id)
    
    # Optional: You might want to also delete related data, like their scans.
    # This line will delete all scans associated with the user's email.
    Scan.query.filter_by(user_email=user_to_delete.email).delete()

    # Delete the user from the database
    db.session.delete(user_to_delete)
    db.session.commit()
    
    flash(f'User {user_to_delete.email} has been deleted successfully.', 'success')
    return redirect('/admin/users')


@app.route('/delete_scan/<int:scan_id>', methods=['POST'])
def delete_scan(scan_id):
    if 'user' not in session:
        return redirect('/')
    scan = Scan.query.get_or_404(scan_id)
    if scan.user_email != session['user']:
        from flask import abort
        abort(403)
    if scan.image_path and os.path.exists(scan.image_path):
        try:
            os.remove(scan.image_path)
        except OSError as e:
            print(f"Error deleting file {scan.image_path}: {e}")
    db.session.delete(scan)
    db.session.commit()
    return redirect('/history')

@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    fb = Feedback(name=request.form["name"], email=request.form["email"], category=request.form["category"], message=request.form["message"])
    db.session.add(fb)
    db.session.commit()
    return redirect("/dashboard")

# --- Admin Dashboard ---
@app.route("/admin_dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect("/admin_login")
    return render_template("admin_dashboard.html")

@app.route("/admin/users")
def admin_users():
    if "admin" not in session:
        return redirect("/admin_login")
    users = User.query.all()
    return render_template("admin_users.html", users=users)

@app.route("/admin/scans")
def admin_scans():
    if "admin" not in session:
        return redirect("/admin_login")
    scans = Scan.query.all()
    return render_template("admin_scans.html", scans=scans)

@app.route("/admin/analytics")
def admin_analytics():
    if "admin" not in session:
        return redirect("/admin_login")
    total_users = User.query.count()
    total_scans = Scan.query.count()
    fake_count = Scan.query.filter(Scan.prediction.like('%Fake%')).count()
    real_count = Scan.query.filter(Scan.prediction.like('%Legitimate%')).count()
    scan_dates = db.session.query(func.date(Scan.timestamp), func.count()).group_by(func.date(Scan.timestamp)).order_by(func.date(Scan.timestamp)).all()
    labels = [d[0] for d in scan_dates]
    values = [d[1] for d in scan_dates]
    return render_template("admin_analytics.html", total_users=total_users, total_scans=total_scans, fake_count=fake_count, real_count=real_count, scan_labels=labels, scan_values=values)

@app.route("/admin/feedback")
def admin_feedback():
    if "admin" not in session:
        return redirect("/admin_login")
    feedbacks = Feedback.query.all()
    return render_template("admin_feedback.html", feedbacks=feedbacks)

@app.route('/admin/reply_feedback/<int:feedback_id>', methods=['POST'])
def reply_to_feedback(feedback_id):
    if 'admin' not in session:
        return redirect('/admin_login')
    feedback = Feedback.query.get_or_404(feedback_id)
    reply_text = request.form.get('reply_message')
    if reply_text:
        feedback.admin_reply = reply_text
        feedback.status = 'Replied'
        db.session.commit()
        try:
            send_feedback_reply_email(feedback)
            flash('Reply sent to the user successfully.', 'success')
        except Exception as e:
            print(f"Failed to send feedback reply email: {e}")
            flash('Reply was saved, but failed to send email notification to the user.', 'error')
    return redirect('/admin/feedback')

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5050)



