# -*- coding: utf-8 -*-

import os
import re
import string
from datetime import datetime, time, timedelta

import joblib
from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
)
from flask_sqlalchemy import SQLAlchemy
from google import genai
from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# =========================================================
# ENVIRONMENT CONFIGURATION
# =========================================================

load_dotenv()

app = Flask(__name__)

secret_key = os.getenv("SECRET_KEY")

if not secret_key:
    raise RuntimeError(
        "SECRET_KEY is missing. Add a persistent SECRET_KEY "
        "to your .env file."
    )

app.secret_key = secret_key

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise RuntimeError(
        "DATABASE_URL is missing. Set it in your .env file."
    )

app.config["SQLALCHEMY_DATABASE_URI"] = database_url

db = SQLAlchemy(app)


# =========================================================
# GEMINI CONFIGURATION
# =========================================================

gemini_api_key = os.getenv("GEMINI_API_KEY")

gemini_client = (
    genai.Client(api_key=gemini_api_key)
    if gemini_api_key
    else None
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
)


def analyze_with_gemini(news_text):
    """
    Analyze an article using Gemini.

    This analysis is not a substitute for independent
    verification of the article's claims.
    """

    if not gemini_client:
        return {
            "success": False,
            "message": (
                "Gemini is not configured. Add GEMINI_API_KEY "
                "to your .env file and restart the application."
            ),
        }

    prompt = f"""
You are an AI assistant supporting a Fake News Detection application.

Analyze the following news article.

IMPORTANT:
- Do not claim that the article is definitively true or false.
- Do not invent facts or sources.
- Explain what the article claims.
- Identify important claims that may require verification.
- Give practical advice about verifying those claims.
- Distinguish analysis from factual verification.

News Article:
{news_text}

Provide your response with these sections:

1. Summary
2. Key Claims
3. Analysis
4. Verification Advice
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        response_text = getattr(response, "text", None)

        if not response_text:
            return {
                "success": False,
                "message": (
                    "Gemini returned an empty response. "
                    "Please try again."
                ),
            }

        return {
            "success": True,
            "message": response_text,
        }

    except Exception:
        app.logger.exception("Gemini API request failed.")

        return {
            "success": False,
            "message": (
                "Gemini AI analysis is temporarily unavailable. "
                "Check your API key, model name, network connection, "
                "and Gemini API quota."
            ),
        }


# =========================================================
# USER MODEL
# =========================================================

class User(db.Model):

    __tablename__ = "users"

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    username = db.Column(
        db.String(50),
        unique=True,
        nullable=False,
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False,
    )

    password_hash = db.Column(
        db.Text,
        nullable=False,
    )

    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now(),
    )

    predictions = db.relationship(
        "PredictionHistory",
        back_populates="user",
        lazy=True,
    )


# =========================================================
# PREDICTION HISTORY MODEL
# =========================================================

class PredictionHistory(db.Model):

    __tablename__ = "prediction_history"

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "users.id",
            name="fk_prediction_history_user",
        ),
        nullable=True,
    )

    news_text = db.Column(
        db.Text,
        nullable=False,
    )

    prediction = db.Column(
        db.String(20),
        nullable=False,
    )

    decision_score = db.Column(
        db.Float,
        nullable=False,
    )

    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now(),
    )

    user = db.relationship(
        "User",
        back_populates="predictions",
    )


# =========================================================
# ADMIN USER MODEL
# =========================================================

class AdminUser(db.Model):

    __tablename__ = "admin_users"

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    username = db.Column(
        db.String(50),
        unique=True,
        nullable=False,
    )

    password_hash = db.Column(
        db.Text,
        nullable=False,
    )

    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now(),
    )


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

MODEL_DIR = os.path.join(app.root_path, "model")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_fake_news_model.pkl",
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "final_tfidf_vectorizer.pkl",
)


def load_ml_artifacts():
    """Load the trained model and TF-IDF vectorizer."""

    missing_files = [
        path
        for path in (MODEL_PATH, VECTORIZER_PATH)
        if not os.path.isfile(path)
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required machine-learning files are missing: "
            + ", ".join(missing_files)
        )

    loaded_model = joblib.load(MODEL_PATH)
    loaded_vectorizer = joblib.load(VECTORIZER_PATH)

    return loaded_model, loaded_vectorizer


model, tfidf = load_ml_artifacts()


# =========================================================
# NLP INITIALIZATION
# =========================================================

lemmatizer = WordNetLemmatizer()

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    app.logger.warning(
        "NLTK stopwords corpus is missing. "
        "Text preprocessing will continue without stopword removal."
    )
    stop_words = set()


def clean_text(text):
    """
    Normalize text for the existing TF-IDF model.

    The preprocessing should match the preprocessing used
    when the model was trained.
    """

    text = str(text or "").lower()

    # Remove URLs.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text,
    )

    # Remove HTML tags.
    text = re.sub(
        r"<.*?>",
        "",
        text,
    )

    # Remove punctuation.
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers.
    text = re.sub(r"\d+", "", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    cleaned_words = []

    for word in words:
        if word in stop_words:
            continue

        try:
            word = lemmatizer.lemmatize(word)
        except LookupError:
            # Continue if the WordNet corpus is not installed.
            pass

        cleaned_words.append(word)

    return " ".join(cleaned_words)


# =========================================================
# TEXT ENCODING FIX
# =========================================================

def fix_text_encoding(text):
    """
    Fix common UTF-8 text incorrectly decoded as Latin-1.

    If the text cannot safely be converted, return it unchanged.
    """

    if not text:
        return text

    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


# =========================================================
# AUTHENTICATION HELPERS
# =========================================================

def is_admin():
    return (
        session.get("role") == "admin"
        and session.get("admin_logged_in") is True
    )


def is_user():
    return (
        session.get("role") == "user"
        and session.get("user_logged_in") is True
        and session.get("user_id") is not None
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        login_as = request.form.get(
            "login_as", ""
        ).strip().lower()

        username = request.form.get(
            "username", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        if login_as not in {"admin", "user"}:
            return render_template(
                "login.html",
                error=(
                    "Please choose whether you are signing in as an "
                    "administrator or normal user."
                ),
                selected_role=login_as,
            )

        if not username or not password:
            return render_template(
                "login.html",
                error="Please enter username and password.",
                selected_role=login_as,
            )

        if login_as == "admin":
            admin = AdminUser.query.filter_by(
                username=username
            ).first()

            if admin and check_password_hash(
                admin.password_hash,
                password,
            ):
                session.clear()

                session["admin_logged_in"] = True
                session["admin_username"] = admin.username
                session["role"] = "admin"

                return redirect(url_for("dashboard"))

        else:
            user = User.query.filter_by(
                username=username
            ).first()

            if user and check_password_hash(
                user.password_hash,
                password,
            ):
                session.clear()

                session["user_logged_in"] = True
                session["user_id"] = user.id
                session["username"] = user.username
                session["role"] = "user"

                return redirect(url_for("user_dashboard"))

        return render_template(
            "login.html",
            error="Invalid username or password for the selected account type.",
            selected_role=login_as,
        )

    success = None

    if request.args.get("registered") == "1":
        success = "Registration successful! Please log in."

    return render_template(
        "login.html",
        success=success,
        selected_role="",
    )


# =========================================================
# USER REGISTRATION
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get(
            "username", ""
        ).strip()

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        confirm_password = request.form.get(
            "confirm_password", ""
        )

        if not username or not email or not password:
            return render_template(
                "register.html",
                error="Please fill in all required fields.",
            )

        if len(username) < 3 or len(username) > 50:
            return render_template(
                "register.html",
                error="Username must be between 3 and 50 characters.",
            )

        if not re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            email,
        ):
            return render_template(
                "register.html",
                error="Please enter a valid email address.",
            )

        if len(password) < 8:
            return render_template(
                "register.html",
                error="Password must contain at least 8 characters.",
            )

        if password != confirm_password:
            return render_template(
                "register.html",
                error="Passwords do not match.",
            )

        if User.query.filter_by(username=username).first():
            return render_template(
                "register.html",
                error="This username is already registered.",
            )

        if User.query.filter_by(email=email).first():
            return render_template(
                "register.html",
                error="This email address is already registered.",
            )

        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
        )

        try:
            db.session.add(new_user)
            db.session.commit()

        except Exception:
            db.session.rollback()
            app.logger.exception("User registration failed.")

            return render_template(
                "register.html",
                error=(
                    "Registration could not be completed. "
                    "Please try again."
                ),
            )

        return redirect(
            url_for("login", registered="1")
        )

    return render_template("register.html")


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# =========================================================
# ADMIN HISTORY - FIXED
# =========================================================

@app.route("/history", methods=["GET"])
def history():

    # Only admins can access all users' prediction history.
    if not is_admin():
        return redirect(url_for("login"))

    # Read the filter values used by history.html.
    search = request.args.get(
        "search", ""
    ).strip()

    prediction_filter = request.args.get(
        "prediction", "ALL"
    ).strip()

    date_from = request.args.get(
        "date_from", ""
    ).strip()

    date_to = request.args.get(
        "date_to", ""
    ).strip()

    page = request.args.get(
        "page", 1, type=int
    )

    if page is None or page < 1:
        page = 1

    # Start with all prediction records.
    query = PredictionHistory.query

    # Search the article text.
    if search:
        query = query.filter(
            PredictionHistory.news_text.ilike(
                f"%{search}%"
            )
        )

    # Filter by prediction result.
    if prediction_filter in ("FAKE NEWS", "REAL NEWS"):
        query = query.filter(
            PredictionHistory.prediction == prediction_filter
        )
    else:
        prediction_filter = "ALL"

    # Filter from the beginning of the selected date.
    if date_from:
        try:
            start_date = datetime.strptime(
                date_from, "%Y-%m-%d"
            ).date()

            start_datetime = datetime.combine(
                start_date, time.min
            )

            query = query.filter(
                PredictionHistory.created_at >= start_datetime
            )

        except ValueError:
            date_from = ""

    # Include the entire ending date.
    if date_to:
        try:
            end_date = datetime.strptime(
                date_to, "%Y-%m-%d"
            ).date()

            next_day = end_date + timedelta(days=1)

            end_datetime = datetime.combine(
                next_day, time.min
            )

            query = query.filter(
                PredictionHistory.created_at < end_datetime
            )

        except ValueError:
            date_to = ""

    # Newest predictions first.
    query = query.order_by(
        PredictionHistory.created_at.desc()
    )

    # Create the pagination object required by history.html.
    pagination = query.paginate(
        page=page,
        per_page=10,
        error_out=False,
    )

    predictions = pagination.items

    return render_template(
        "history.html",
        predictions=predictions,
        history=predictions,
        pagination=pagination,
        search=search,
        prediction_filter=prediction_filter,
        date_from=date_from,
        date_to=date_to,
    )


# =========================================================
# DELETE ONE PREDICTION
# =========================================================

@app.route(
    "/delete_prediction/<int:prediction_id>",
    methods=["POST"],
)
def delete_prediction(prediction_id):

    if not is_admin():
        return redirect(url_for("login"))

    prediction_record = db.session.get(
        PredictionHistory,
        prediction_id,
    )

    if prediction_record is not None:
        try:
            db.session.delete(prediction_record)
            db.session.commit()

        except Exception:
            db.session.rollback()
            app.logger.exception(
                "Could not delete prediction %s.",
                prediction_id,
            )

    # The history endpoint now exists.
    return redirect(url_for("history"))


# =========================================================
# CLEAR ALL PREDICTION HISTORY
# =========================================================

@app.route("/clear_history", methods=["POST"])
def clear_history():

    if not is_admin():
        return redirect(url_for("login"))

    try:
        db.session.query(PredictionHistory).delete()
        db.session.commit()

    except Exception:
        db.session.rollback()
        app.logger.exception("Could not clear prediction history.")

    return redirect(url_for("history"))


# =========================================================
# USER DASHBOARD
# =========================================================

@app.route("/user-dashboard")
def user_dashboard():

    if not is_user():
        return redirect(url_for("login"))

    user = db.session.get(
        User,
        session["user_id"],
    )

    if user is None:
        session.clear()
        return redirect(url_for("login"))

    query = PredictionHistory.query.filter_by(
        user_id=user.id
    )

    total_predictions = query.count()

    fake_count = query.filter_by(
        prediction="FAKE NEWS"
    ).count()

    real_count = query.filter_by(
        prediction="REAL NEWS"
    ).count()

    if total_predictions:
        fake_percentage = round(
            fake_count / total_predictions * 100,
            2,
        )

        real_percentage = round(
            real_count / total_predictions * 100,
            2,
        )

    else:
        fake_percentage = 0
        real_percentage = 0

    recent_predictions = (
        query
        .order_by(PredictionHistory.created_at.desc())
        .limit(10)
        .all()
    )

    return render_template(
        "user_dashboard.html",
        username=user.username,
        total_predictions=total_predictions,
        fake_count=fake_count,
        real_count=real_count,
        fake_percentage=fake_percentage,
        real_percentage=real_percentage,
        recent_predictions=recent_predictions,
    )


# =========================================================
# USER HISTORY
# =========================================================

@app.route("/user-history")
def user_history():

    if not is_user():
        return redirect(url_for("login"))

    user = db.session.get(
        User,
        session["user_id"],
    )

    if user is None:
        session.clear()
        return redirect(url_for("login"))

    predictions = (
        PredictionHistory.query
        .filter_by(user_id=user.id)
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )

    return render_template(
        "user_history.html",
        username=user.username,
        predictions=predictions,
    )


# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    if not is_admin():
        return redirect(url_for("login"))

    users = User.query.order_by(
        User.created_at.desc()
    ).all()

    total_users = len(users)

    predictions = (
        PredictionHistory.query
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )

    total_predictions = len(predictions)

    fake_count = sum(
        1 for item in predictions
        if item.prediction == "FAKE NEWS"
    )

    real_count = sum(
        1 for item in predictions
        if item.prediction == "REAL NEWS"
    )

    if total_predictions:
        fake_percentage = round(
            fake_count / total_predictions * 100,
            2,
        )

        real_percentage = round(
            real_count / total_predictions * 100,
            2,
        )

    else:
        fake_percentage = 0
        real_percentage = 0

    user_stats = []

    for user in users:

        user_predictions = (
            PredictionHistory.query
            .filter_by(user_id=user.id)
            .all()
        )

        user_fake_count = sum(
            1 for item in user_predictions
            if item.prediction == "FAKE NEWS"
        )

        user_real_count = sum(
            1 for item in user_predictions
            if item.prediction == "REAL NEWS"
        )

        user_stats.append({
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at,
            "total_predictions": len(user_predictions),
            "fake_count": user_fake_count,
            "real_count": user_real_count,
        })

    recent_predictions = predictions[:10]

    return render_template(
        "dashboard.html",
        total_users=total_users,
        users=users,
        user_stats=user_stats,
        total_predictions=total_predictions,
        fake_count=fake_count,
        real_count=real_count,
        fake_percentage=fake_percentage,
        real_percentage=real_percentage,
        recent_predictions=recent_predictions,
    )


# =========================================================
# GEMINI TEST
# =========================================================

@app.route("/gemini-test")
def gemini_test():

    if not is_admin():
        return redirect(url_for("login"))

    sample_news = (
        "The government has announced that every college "
        "student will receive a free laptop this year."
    )

    analysis = analyze_with_gemini(sample_news)

    return render_template(
        "gemini_test.html",
        analysis=analysis,
        model_name=GEMINI_MODEL,
        key_configured=gemini_client is not None,
        sample_news=sample_news,
    )


# =========================================================
# NEWS PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    news_text = request.form.get(
        "news_text", ""
    ).strip()

    news_text = fix_text_encoding(news_text)

    if not news_text:
        return render_template(
            "index.html",
            error="Please enter a news article.",
        )

    try:
        cleaned_text = clean_text(news_text)

        text_vector = tfidf.transform(
            [cleaned_text]
        )

        prediction_value = model.predict(
            text_vector
        )[0]

        # Keep compatibility with your original binary model:
        # 0 = FAKE NEWS; any other label = REAL NEWS.
        if prediction_value == 0:
            result = "FAKE NEWS"
        else:
            result = "REAL NEWS"

        # Obtain the decision score when the model supports it.
        if hasattr(model, "decision_function"):
            raw_score = model.decision_function(text_vector)

            # The prediction uses one article at a time.
            decision_score = float(raw_score[0])
        else:
            # Models without decision_function do not provide
            # the original SVM decision score.
            decision_score = 0.0

    except Exception:
        app.logger.exception("News prediction failed.")

        return render_template(
            "index.html",
            error=(
                "The article could not be analyzed. "
                "Please try again or check the model files."
            ),
        )

    # Associate the prediction with a logged-in normal user.
    current_user_id = None

    if is_user():
        current_user_id = session.get("user_id")

    prediction_record = PredictionHistory(
        user_id=current_user_id,
        news_text=news_text,
        prediction=result,
        decision_score=decision_score,
    )

    try:
        db.session.add(prediction_record)
        db.session.commit()

    except Exception:
        db.session.rollback()
        app.logger.exception("Could not save prediction history.")

        return render_template(
            "index.html",
            error=(
                "The prediction was generated, but it could not "
                "be saved. Check your database connection and schema."
            ),
        )

    return render_template(
        "result.html",
        result=result,
        score=round(decision_score, 4),
        news_text=news_text,
    )


# =========================================================
# GEMINI ARTICLE ANALYSIS
# =========================================================

@app.route("/gemini-analyze", methods=["POST"])
def gemini_analyze():

    news_text = request.form.get(
        "news_text", ""
    ).strip()

    news_text = fix_text_encoding(news_text)

    if not news_text:
        return redirect(url_for("home"))

    analysis = analyze_with_gemini(news_text)

    return render_template(
        "gemini_result.html",
        news_text=news_text,
        analysis=analysis,
    )


# =========================================================
# CREATE ADMIN USER
# =========================================================

def create_admin():

    username = os.getenv("ADMIN_USERNAME", "admin").strip()
    password = os.getenv("ADMIN_PASSWORD")

    if not username:
        raise RuntimeError(
            "ADMIN_USERNAME cannot be empty."
        )

    existing_admin = AdminUser.query.filter_by(
        username=username
    ).first()

    if existing_admin:
        print("Admin user already exists.")
        return

    # Do not create a new admin with a predictable default password.
    if not password:
        raise RuntimeError(
            "ADMIN_PASSWORD is missing. Add ADMIN_USERNAME and "
            "a strong ADMIN_PASSWORD to your .env file before "
            "creating the first admin account."
        )

    admin = AdminUser(
        username=username,
        password_hash=generate_password_hash(password),
    )

    try:
        db.session.add(admin)
        db.session.commit()
        print("Admin user created successfully.")

    except Exception:
        db.session.rollback()
        app.logger.exception("Admin user creation failed.")
        raise


# =========================================================
# INITIALIZE DATABASE AND RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    with app.app_context():

        # Creates missing tables but does NOT migrate existing tables.
        db.create_all()

        create_admin()

    debug_enabled = (
        os.getenv("FLASK_DEBUG", "false").lower()
        in {"1", "true", "yes"}
    )

    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=debug_enabled,
    )
