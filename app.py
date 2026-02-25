from flask import Flask, render_template, request, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import joblib
import pandas as pd
import os

# ================= APP CONFIG =================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "ckd_super_secure_2026")

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ckd_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ================= DATABASE MODELS =================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(300), nullable=False)


class TestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    age = db.Column(db.Float)
    blood_pressure = db.Column(db.Float)
    specific_gravity = db.Column(db.Float)
    albumin = db.Column(db.Float)
    sugar = db.Column(db.Float)
    red_blood_cells = db.Column(db.Integer)
    pus_cell = db.Column(db.Integer)
    #pus_cell_clumps = db.Column(db.Integer)
    #bacteria = db.Column(db.Integer)
    blood_glucose_random = db.Column(db.Float)
    blood_urea = db.Column(db.Float)
    serum_creatinine = db.Column(db.Float)
    #sodium = db.Column(db.Float)
    #potassium = db.Column(db.Float)
    haemoglobin = db.Column(db.Float)
    packed_cell_volume = db.Column(db.Float)
    #white_blood_cell_count = db.Column(db.Float)
    red_blood_cell_count = db.Column(db.Float)
    hypertension = db.Column(db.Integer)
    diabetes_mellitus = db.Column(db.Integer)
    #coronary_artery_disease = db.Column(db.Integer)
    #appetite = db.Column(db.Integer)
    #peda_edema = db.Column(db.Integer)
    #aanemia = db.Column(db.Integer)

    prediction = db.Column(db.String(50))
    prob_ckd = db.Column(db.Float)
    prob_normal = db.Column(db.Float)


with app.app_context():
    db.create_all()

# ================= LOAD MODEL SAFELY =================
MODEL_PATH = os.path.join("models", "adaboost_kidney_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found inside models/ folder")

model = joblib.load(MODEL_PATH)

# ================= LOGIN REQUIRED DECORATOR =================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated_function

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username")
    password = request.form.get("password")

    if User.query.filter_by(username=username).first():
        return render_template("home.html", error="Username already exists")

    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password=hashed_password)

    db.session.add(new_user)
    db.session.commit()

    return redirect(url_for("home"))


@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")

    user = User.query.filter_by(username=username).first()

    if user and check_password_hash(user.password, password):
        session["user_id"] = user.id
        session["username"] = user.username
        return redirect(url_for("test_page"))

    return render_template("home.html", error="Invalid Credentials")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/test")
@login_required
def test_page():
    return render_template("test.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        input_data = {
            'age': float(request.form['age']),
            'blood_pressure': float(request.form['bp']),
            'specific_gravity': float(request.form['sg']),
            'albumin': float(request.form['al']),
            'sugar': float(request.form['su']),
            'red_blood_cells': int(request.form['rbc']),
            'pus_cell': int(request.form['pc']),
            #'pus_cell_clumps': int(request.form['pcc']),
            #'bacteria': int(request.form['ba']),
            'blood_glucose_random': float(request.form['bgr']),
            'blood_urea': float(request.form['bu']),
            'serum_creatinine': float(request.form['sc']),
            #'sodium': float(request.form['sod']),
            #'potassium': float(request.form['pot']),
            'haemoglobin': float(request.form['hg']),
            'packed_cell_volume': float(request.form['pcv']),
            #'white_blood_cell_count': float(request.form['wbcc']),
            'red_blood_cell_count': float(request.form['rbcc']),
            'hypertension': int(request.form['htn']),
            'diabetes_mellitus': int(request.form['dm']),
            #'coronary_artery_disease': int(request.form['cad']),
            #'appetite': int(request.form['app']),
            #'peda_edema': int(request.form['pe']),
            #'aanemia': int(request.form['ane'])
        }

        df = pd.DataFrame([input_data])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)

        result = "Kidney Disease Detected" if prediction == 0 else "No Kidney Disease"
        prob_ckd = round(probability[0][0] * 100, 2)
        prob_normal = round(probability[0][1] * 100, 2)

        new_result = TestResult(
            user_id=session["user_id"],
            **input_data,
            prediction=result,
            prob_ckd=prob_ckd,
            prob_normal=prob_normal
        )

        db.session.add(new_result)
        db.session.commit()

        return render_template(
            "result.html",
            result=result,
            prob_ckd=prob_ckd,
            prob_normal=prob_normal
        )

    except Exception as e:
        return f"Prediction Error: {str(e)}"


@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    latest = TestResult.query.filter_by(user_id=user_id).order_by(TestResult.id.desc()).first()

    if not latest:
        return render_template("dashboard.html", message="No test history available")

    return render_template("dashboard.html", data=latest)


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)