from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from functools import lru_cache
from datetime import datetime
import requests
import PyPDF2
import re
import pandas as pd
import joblib
import spacy
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# --- DATABASE & SECURITY CONFIG ---
app.config['SECRET_KEY'] = 'your_super_secret_key_12345'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    history = db.relationship('AnalysisHistory', backref='user', lazy=True)
    jobs = db.relationship('SavedJob', backref='user', lazy=True)

class AnalysisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    target_job = db.Column(db.String(100), nullable=False)
    ats_score = db.Column(db.Integer, nullable=False)
    low_lpa = db.Column(db.Float, nullable=False)
    high_lpa = db.Column(db.Float, nullable=False)
    date_analyzed = db.Column(db.DateTime, default=datetime.utcnow)

class SavedJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(150), nullable=False)
    company = db.Column(db.String(150), nullable=False)
    url = db.Column(db.String(500), nullable=False)
    status = db.Column(db.String(50), default="Saved") # Saved, Applied, Interviewing, Rejected
    date_saved = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

# --- ML & NLP SETUP ---
MODEL_PATH = 'salary_regressor_joblib.pkl'
try:
    ml_dict = joblib.load(MODEL_PATH)
    ml_model = ml_dict['model']
except Exception:
    ml_model = None
    print(f"⚠️ Warning: Could not load {MODEL_PATH}. Using fallback math.")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

TECH_SKILLS_DB = [
    # Programming Languages
    "python", "java", "c++", "c#", "c", "javascript", "typescript", "ruby", "php", "swift", "kotlin", "go", "rust", "dart", "r", "scala",
    # Web & Mobile Development
    "html", "css", "react", "react native", "node.js", "django", "flask", "fastapi", "spring boot", "angular", "vue.js", "next.js", "express.js", "tailwind", "bootstrap", "flutter", "android", "ios",
    # AI, ML & Data
    "machine learning", "deep learning", "nlp", "ai", "artificial intelligence", "computer vision", "data science", "tensorflow", "keras", "pytorch", "pandas", "numpy", "scikit-learn", "langchain", "llm", "generative ai", "prompt engineering", "openai", "hugging face", "opencv", "matplotlib", "seaborn",
    # Databases & Cloud
    "sql", "nosql", "mongodb", "postgresql", "mysql", "sqlite", "redis", "docker", "aws", "azure", "gcp", "git", "kubernetes", "linux", "ci/cd", "jenkins", "terraform",
    # Business, Design, Academia & Logistics
    "agile", "scrum", "jira", "ui/ux", "figma", "data analysis", "tableau", "powerbi", "excel", "teaching", "lesson planning", "curriculum development", "public speaking", "student engagement", "driving", "logistics", "route planning", "leadership", "team management", "project management", "customer service", "data entry", "scheduling"
]

if nlp:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("TECH_SKILLS", list(nlp.pipe(TECH_SKILLS_DB)))

def extract_skills_nlp(text):
    if not nlp: return [] 
    doc = nlp(text)
    return sorted(list(set([doc[start:end].text.title() for match_id, start, end in matcher(doc)])))

# --- LOGIC FUNCTIONS ---
def analyze_resume_locally(text, location="Tier 2/3 (Other Cities)", manual_data=None):
    if manual_data:
        skills = [s.strip().title() for s in manual_data['skills'].split(',') if s.strip()]
        exp_years = int(manual_data['experience'])
        degree = manual_data['degree']
    else:
        skills = extract_skills_nlp(text)
        exp_match = re.search(r"(\d+)\s*(?:years?|yrs?)", text, re.I)
        exp_years = int(exp_match.group(1)) if exp_match else 0
        degree_match = re.search(r"\b(Ph\.?D|Doctorate|M\.?Tech|Master.?of.?Technology|M\.?E|B\.?Tech|B\.?E|Bachelor.?of.?Technology|B\.?Sc|M\.?Sc|BCA|MCA|Diploma|12th|10th)\b", text, re.I)
        degree = degree_match.group(1).replace('.', '').strip().upper() if degree_match else "Not Mentioned"
        degree = {"BTECH": "B.Tech", "BE": "B.E", "MTECH": "M.Tech", "ME": "M.E", "MSC": "M.Sc", "BSC": "B.Sc"}.get(degree, degree)

    base_salary_lpa = 0.0
    if ml_model is not None:
        census_edu = {"B.Tech": "Bachelors", "M.Tech": "Masters", "Ph.D": "Doctorate", "MCA": "Masters", "Diploma": "Some-college"}.get(degree, "Bachelors")
        edu_num = {"Bachelors": 13, "Masters": 14, "Doctorate": 16, "Some-college": 10}.get(census_edu, 13)
        input_data = pd.DataFrame([{'age': 22 + exp_years, 'education-num': edu_num, 'experience': exp_years, 'workclass': 'Private', 'education': census_edu, 'occupation': 'Prof-specialty', 'marital-status': 'Never-married', 'relationship': 'Not-in-family', 'race': 'Asian-Pac-Islander', 'sex': 'Male', 'native-country': 'India'}])
        base_salary_lpa = (ml_model.predict(input_data)[0] / 100000.0) + (len(skills) * 0.3)
    else:
        base_salary_lpa = 3.0 + (exp_years * 0.8) + (len(skills) * 0.4)
        if degree in ["B.Tech", "B.E", "MCA"]: base_salary_lpa += 2.0
        elif degree in ["M.Tech", "M.E"]: base_salary_lpa += 3.5

    if "Tier 1" in location: base_salary_lpa *= 1.4  
    elif "Remote" in location: base_salary_lpa *= 1.6  
    
    return {"degree": degree, "experience": exp_years, "skills": skills, "low_lpa": round(base_salary_lpa * 0.85, 1), "high_lpa": round(base_salary_lpa * 1.15, 1), "location": location}

JOB_SKILLS_DB = {
    "ai engineer": "python machine learning deep learning tensorflow pytorch nlp computer vision neural networks sql git docker aws pandas scikit-learn langchain llm",
    "data scientist": "python sql r pandas numpy scikit-learn machine learning statistics data visualization tableau powerbi excel hadoop spark",
    "data analyst": "sql excel tableau powerbi python pandas data visualization statistics mathematics communication",
    "frontend developer": "html css javascript typescript react vue.js angular tailwind bootstrap git ui/ux figma",
    "backend developer": "python java c# node.js django flask spring boot sql postgresql mongodb docker aws rest api",
    "full stack developer": "html css javascript react node.js python sql mongodb docker git aws typescript express.js",
    "mobile developer": "swift kotlin flutter dart react native ios android java git ui/ux",
    "cloud engineer": "aws azure gcp linux docker kubernetes terraform bash ci/cd git python networking",
    "devops engineer": "linux bash docker kubernetes jenkins ci/cd aws azure terraform ansible git python",
    "cybersecurity analyst": "linux networking python bash ethical hacking cryptography security firewalls c sql",
    "ui/ux designer": "figma adobe xd photoshop illustrator wireframing prototyping user research html css",
    "product manager": "agile scrum jira project management communication leadership roadmap strategy ui/ux data analysis",
    "digital marketer": "seo sem content marketing google analytics social media marketing copywriting email marketing excel",
    "software developer": "java python c++ c# javascript react node.js html css sql docker git aws",
    "teacher": "teaching lesson planning curriculum development public speaking student engagement classroom management communication empathy patience grading",
    "lecturer": "academic research higher education public speaking curriculum development mentoring publishing presentation skills subject matter expertise",
    "professor": "academic research higher education public speaking curriculum development mentoring publishing presentation skills subject matter expertise leadership",
    "driver": "driving logistics route planning vehicle maintenance time management safety regulations customer service navigation",
    "delivery driver": "driving logistics route planning time management safety regulations customer service navigation inventory management",
    "manager": "leadership team management budgeting conflict resolution negotiation project management communication strategic planning operations",
    "project manager": "project management agile scrum jira leadership team management communication budgeting stakeholder management",
    "receptionist": "customer service scheduling office administration data entry communication multi-tasking phone etiquette organization microsoft office",
    "administrative assistant": "scheduling office administration data entry communication organization microsoft office record keeping typing",
    "travel agent": "travel booking itinerary planning customer service geography sales negotiation hospitality communication booking software tourism",
    "default": "communication problem solving teamwork time management leadership critical thinking organization adaptability"
}

def generate_roadmap(current_skills, target_job):
    matched_profile = next((k for k in JOB_SKILLS_DB if k in target_job.lower()), "default")
    ideal_skills_set = set(JOB_SKILLS_DB[matched_profile].split())
    current_skills_set = set([s.lower() for s in current_skills])
    try:
        tfidf_matrix = TfidfVectorizer().fit_transform([JOB_SKILLS_DB[matched_profile], " ".join([s.lower() for s in current_skills])])
        match_score = min(int((cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100) + 10), 100) 
    except:
        match_score = 15
    return {"score": match_score, "missing": [s.title() for s in list(ideal_skills_set - current_skills_set)[:5]]}

def generate_interview_prep(target_job, missing_skills):
    questions = [
        {"q": f"Why do you want to work as a {target_job.title()}?", "tip": "Focus on your passion for the industry and how your current skills translate."},
        {"q": "Tell me about a time you solved a complex problem.", "tip": "Use the STAR method (Situation, Task, Action, Result)."},
    ]
    if missing_skills:
        questions.append({"q": f"We use {missing_skills[0]} extensively. How would you quickly get up to speed?", "tip": "Mention your proven ability to learn new tools quickly and cite a specific plan."})
    if len(missing_skills) > 1:
        questions.append({"q": f"How does your background compensate for your lack of direct experience with {missing_skills[1]}?", "tip": "Highlight related fundamental concepts that transfer over."})
    else:
        questions.append({"q": "Where do you see your technical skills growing in the next year?", "tip": "Align your learning goals with the company's stack."})
    
    questions.append({"q": "Do you have any questions for us?", "tip": "Always prepare 2-3 questions about the team culture or specific projects."})
    return questions

@lru_cache(maxsize=32)
def fetch_live_jobs(search_term, location="India"):
    search_loc = "Remote" if "Remote" in location else ("Bangalore, India" if "Tier 1" in location else "India")
    core_keyword = search_term.split('/')[0].strip()
    url = "https://jsearch.p.rapidapi.com/search"
    querystring = {"query": f"{core_keyword} in {search_loc}", "page": "1", "num_pages": "1", "date_posted": "month"}
    headers = {"X-RapidAPI-Key": "c19f55d67bmsh8adf8b876f3c023p1586b5jsn6e8601a3a94c", "X-RapidAPI-Host": "jsearch.p.rapidapi.com"}
    try:
        response = requests.get(url, headers=headers, params=querystring)
        return [{'title': j.get('job_title', 'Unknown'), 'company': j.get('employer_name', 'Unknown'), 'url': j.get('job_apply_link', '#')} for j in response.json().get('data', [])[:6]]
    except: return []

# --- ROUTES ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        hashed_pw = bcrypt.generate_password_hash(request.form.get('password')).decode('utf-8')
        db.session.add(User(username=request.form.get('username'), email=request.form.get('email'), password=hashed_pw))
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email')).first()
        if user and bcrypt.check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user_history = AnalysisHistory.query.filter_by(user_id=current_user.id).order_by(AnalysisHistory.date_analyzed.desc()).all()
    saved_jobs = SavedJob.query.filter_by(user_id=current_user.id).order_by(SavedJob.date_saved.desc()).all()
    return render_template('dashboard.html', history=user_history, saved_jobs=saved_jobs)

@app.route('/save_job', methods=['POST'])
@login_required
def save_job():
    db.session.add(SavedJob(user_id=current_user.id, title=request.form.get('title'), company=request.form.get('company'), url=request.form.get('url')))
    db.session.commit()
    flash('Job saved to your Tracker!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/update_job/<int:job_id>', methods=['POST'])
@login_required
def update_job(job_id):
    job = SavedJob.query.get_or_404(job_id)
    if job.user_id == current_user.id:
        job.status = request.form.get('status')
        db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/delete_job/<int:job_id>', methods=['POST'])
@login_required
def delete_job(job_id):
    job = SavedJob.query.get_or_404(job_id)
    if job.user_id == current_user.id:
        db.session.delete(job)
        db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/analyze', methods=['POST'])
def analyze():
    target = request.form.get('target_job')
    input_method = request.form.get('input_method') 
    location = request.form.get('location')
    
    if input_method == 'upload':
        resume_file = request.files.get('resume_pdf')
        if resume_file and resume_file.filename.lower().endswith('.pdf'):
            resume_text = "".join([page.extract_text() for page in PyPDF2.PdfReader(resume_file).pages if page.extract_text()])
            local_data = analyze_resume_locally(resume_text, location=location)
        else:
            flash('Error: Please provide a valid PDF file', 'danger')
            return redirect(url_for('home'))
    else:
        local_data = analyze_resume_locally("", location=location, manual_data={'degree': request.form.get('manual_degree'), 'experience': request.form.get('manual_experience'), 'skills': request.form.get('manual_skills')})

    roadmap = generate_roadmap(local_data['skills'], target)
    job_listings = fetch_live_jobs(target, location)
    interview_prep = generate_interview_prep(target, roadmap['missing'])

    if current_user.is_authenticated:
        db.session.add(AnalysisHistory(user_id=current_user.id, target_job=target, ats_score=roadmap['score'], low_lpa=local_data['low_lpa'], high_lpa=local_data['high_lpa']))
        db.session.commit()
    
    return render_template('index.html', local_data=local_data, roadmap=roadmap, jobs=job_listings, target=target, interview_prep=interview_prep)

if __name__ == '__main__':
    app.run(debug=True, port=5000)