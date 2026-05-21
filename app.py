"""
app.py
Flask web application for the News Popularity Prediction System.

Only Flask is an external package.
All ML, preprocessing, and database code is implemented from scratch.

Run:
    python app.py
Open: http://127.0.0.1:5000

Default admin login:
    Username: admin
    Password: admin123
"""

import os
import sqlite3
from datetime import datetime
from flask import (Flask, render_template, request,
                   redirect, url_for, session, flash)

from model_io          import load_model
from text_preprocessor import extract_text_features
from data_utils        import normalize_single

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "np_secret_2024_tribhuvan"

MODEL_PATH = "model.pkl"
DB_PATH    = "news_popularity.db"

# ── Load model once at startup ────────────────────────────────────────────────
_model, _feature_stats = None, None
if os.path.exists(MODEL_PATH):
    _model, _feature_stats = load_model(MODEL_PATH)
else:
    print(f"[WARNING] '{MODEL_PATH}' not found. Run train.py first.")


# ── Database ──────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    NOT NULL UNIQUE,
            email    TEXT    NOT NULL UNIQUE,
            password TEXT    NOT NULL,
            role     TEXT    NOT NULL DEFAULT 'user'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            title      TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            num_hrefs  INTEGER NOT NULL DEFAULT 0,
            num_imgs   INTEGER NOT NULL DEFAULT 0,
            num_videos INTEGER NOT NULL DEFAULT 0,
            prediction TEXT    NOT NULL,
            created_at TEXT    NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    cur.execute("SELECT id FROM users WHERE username='admin'")
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users (username, email, password, role) VALUES (?,?,?,?)",
            ("admin", "admin@newspred.com", hash_password("admin123"), "admin")
        )
    conn.commit()
    conn.close()


def hash_password(password):
    """
    Simple deterministic hash - no external library.
    For academic purposes only; use bcrypt in production.
    """
    salted = password + "tribhuvan_np_salt_9274"
    total  = 5381
    for ch in salted:
        total = ((total << 5) + total + ord(ch)) & 0xFFFFFFFF
    return str(total)


# ── Route helpers ─────────────────────────────────────────────────────────────

def logged_in():
    return "user_id" in session


def is_admin():
    return session.get("role") == "admin"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("predict") if logged_in() else url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if logged_in():
        return redirect(url_for("predict"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email",    "").strip()
        password = request.form.get("password", "")
        if not username or not email or not password:
            error = "All fields are required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        else:
            conn = get_db()
            conn.execute("SELECT id FROM users WHERE username=? OR email=?",
                         (username, email))
            if conn.execute("SELECT id FROM users WHERE username=? OR email=?",
                            (username, email)).fetchone():
                error = "Username or email already registered."
            else:
                conn.execute(
                    "INSERT INTO users (username, email, password) VALUES (?,?,?)",
                    (username, email, hash_password(password))
                )
                conn.commit()
                conn.close()
                flash("Account created! Please log in.", "success")
                return redirect(url_for("login"))
            conn.close()
    return render_template("register.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    if logged_in():
        return redirect(url_for("predict"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, hash_password(password))
        ).fetchone()
        conn.close()
        if user:
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            return redirect(url_for("predict"))
        else:
            error = "Incorrect username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not logged_in():
        return redirect(url_for("login"))
    result, error = None, None
    if request.method == "POST":
        title      = request.form.get("title",      "").strip()
        content    = request.form.get("content",    "").strip()
        num_hrefs  = int(request.form.get("num_hrefs",  "0") or 0)
        num_imgs   = int(request.form.get("num_imgs",   "0") or 0)
        num_videos = int(request.form.get("num_videos", "0") or 0)

        if not title or not content:
            error = "Please enter both a title and content."
        elif _model is None:
            error = "Model not loaded. Run train.py first."
        else:
            x_raw  = extract_text_features(title, content, num_hrefs, num_imgs, num_videos)
            x_norm = normalize_single(x_raw, _feature_stats)
            result = _model.predict_one(x_norm)

            conn = get_db()
            conn.execute(
                "INSERT INTO predictions "
                "(user_id, title, content, num_hrefs, num_imgs, num_videos, prediction, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (session["user_id"], title, content,
                 num_hrefs, num_imgs, num_videos, result,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            conn.close()

    return render_template("predict.html", result=result, error=error)


@app.route("/history")
def history():
    if not logged_in():
        return redirect(url_for("login"))
    conn = get_db()
    rows = conn.execute(
        "SELECT title, prediction, created_at FROM predictions "
        "WHERE user_id=? ORDER BY id DESC LIMIT 50",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return render_template("history.html", predictions=rows)


@app.route("/admin")
def admin_dashboard():
    if not is_admin():
        return redirect(url_for("predict"))
    conn       = get_db()
    users      = conn.execute("SELECT id, username, email, role FROM users ORDER BY id").fetchall()
    total_pred = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    recent     = conn.execute(
        "SELECT u.username, p.title, p.prediction, p.created_at "
        "FROM predictions p JOIN users u ON p.user_id=u.id "
        "ORDER BY p.id DESC LIMIT 10"
    ).fetchall()
    conn.close()
    return render_template("admin.html", users=users, total_pred=total_pred, recent=recent)


@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
def delete_user(user_id):
    if not is_admin():
        return redirect(url_for("predict"))
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id=? AND role!='admin'", (user_id,))
    conn.commit()
    conn.close()
    flash("User deleted.", "success")
    return redirect(url_for("admin_dashboard"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)