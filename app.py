
from flask import Flask, render_template, request, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

# Ensure instance folder exists
os.makedirs(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance'), exist_ok=True)

# SQLite DB setup
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'site.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Route: Home
@app.route("/")
def index():
    return render_template("index.html")

# Route: Admin Login
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == "admin" and password == "admin123":
            session["admin"] = username
            return redirect("/dashboard")
        return "Invalid admin credentials!"
    return render_template("admin_login.html")

# Route: User Login
@app.route("/user_login", methods=["GET", "POST"])
def user_login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session["user"] = user.email
            return redirect("/dashboard")
        return "Invalid user credentials!"
    return render_template("user_login.html")

# Route: Register
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = generate_password_hash(request.form["password"])
        if User.query.filter_by(email=email).first():
            return "User already exists!"
        new_user = User(name=name, email=email, phone=phone, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect("/user_login")
    return render_template("register.html")

# Route: Dashboard
@app.route("/dashboard")
def dashboard():
    if "user" not in session and "admin" not in session:
        return redirect("/")
    return render_template("dashboard.html", user=session.get("user", None), admin=session.get("admin", None))

# Route: Logout
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")
    
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True,port=5001))

