import streamlit as st
import docx2txt
import PyPDF2
import re
import os
import io
import math
import tempfile
import base64

# NLP & ML
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import collections

# Visualization & PDF
import matplotlib.pyplot as plt
import pandas as pd

# Attempt optional libs
try:
    import spacy
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

try:
    from fpdf import FPDF
    _HAS_FPDF = True
except Exception:
    _HAS_FPDF = False

# ------------------ SETUP / DOWNLOADS ------------------------
nltk_downloads = ["stopwords", "punkt", "averaged_perceptron_tagger"]
for res in nltk_downloads:
    try:
        nltk.data.find(f"corpora/{res}")
    except Exception:
        nltk.download(res)

try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set()

# If spaCy is available but model missing, try to download small model
if _HAS_SPACY:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        try:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            _HAS_SPACY = False
            nlp = None
else:
    nlp = None

# ---------- Helpful skill list (extendable) ----------
COMMON_SKILLS = set([
    "python","java","c++","c","c#","javascript","react","angular","django","flask",
    "sql","mysql","postgresql","mongodb","aws","azure","gcp","docker","kubernetes",
    "tensorflow","pytorch","nlp","machine learning","deep learning","scikit-learn",
    "pandas","numpy","matplotlib","excel","powerbi","tableau","git","linux",
    "rest api","graphql","html","css","bootstrap","node.js","express"
])

# ------------------ UTIL FUNCTIONS -------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    # replace common separators with newline for better section detection
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # keep letters, numbers, newline and spaces and some punctuation
    text = re.sub(r'[^a-z0-9\s\-\n\.\,]', ' ', text)
    # collapse spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_resume_text(uploaded_file):
    if uploaded_file is None:
        return ""
    filetype = getattr(uploaded_file, "type", "")
    name = getattr(uploaded_file, "name", "").lower()
    try:
        if "pdf" in filetype or name.endswith(".pdf"):
            # PyPDF2 PdfReader can accept file-like objects
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        elif "word" in filetype or name.endswith(".docx") or name.endswith(".doc"):
            # docx2txt expects a path -> write to temp
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
            tfile.write(uploaded_file.read())
            tfile.flush()
            tfile.close()
            text = docx2txt.process(tfile.name)
            try:
                os.unlink(tfile.name)
            except Exception:
                pass
            return text
        else:
            # assume text
            content = uploaded_file.read()
            if isinstance(content, bytes):
                return content.decode(errors="ignore")
            return str(content)
    except Exception as e:
        return f"Error extracting resume: {e}"

def calculate_similarity(resume_text, job_desc):
    documents = [resume_text, job_desc]
    tfidf = TfidfVectorizer(stop_words='english')
    try:
        matrix = tfidf.fit_transform(documents)
        score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return round(score * 100, 2)
    except Exception:
        return 0.0

def top_n_tfidf_terms(text, n=25):
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    try:
        X = vec.fit_transform([text])
    except Exception:
        return []
    names = vec.get_feature_names_out()
    sums = X.toarray().sum(axis=0)
    data = list(zip(names, sums))
    data.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in data[:n]]

def extract_skills(resume_text):
    text = resume_text.lower() if resume_text else ""
    found = set()
    # match against COMMON_SKILLS (word boundaries)
    for skill in COMMON_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found.add(skill)

    # simple regex for technologies like 'react js', 'node js', 'c++'
    tech_patterns = [r'\breact\b', r'\bnode\.?js\b', r'c\+\+', r'c#', r'\bsql\b', r'\bnosql\b']
    for pat in tech_patterns:
        m = re.search(pat, text)
        if m:
            found.add(m.group(0))

    # spaCy noun_chunks + entities if available
    if _HAS_SPACY and nlp and resume_text:
        try:
            doc = nlp(resume_text)
            for ent in doc.ents:
                if ent.label_.lower() in ("product","org","skill","technology"):
                    found.add(ent.text.lower())
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.strip().lower()
                if 2 <= len(chunk_text.split()) <= 3:
                    if any(w for w in chunk_text.split() if w not in STOP_WORDS):
                        found.add(chunk_text)
        except Exception:
            pass

    found = set([f for f in found if len(f) > 1])
    return sorted(found)

def fuzzy_match_skill(word, candidates, cutoff=0.8):
    matches = get_close_matches(word, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def parse_sections(resume_text):
    text = resume_text.replace('\r\n', '\n')
    headings = [
        "objective","summary","experience","work experience","professional experience",
        "education","skills","projects","certifications","achievements","internship","contact"
    ]
    try:
        lines = text.splitlines()
        curr = "header"
        sections = {curr: ""}
        for line in lines:
            stripped = line.strip()
            low = stripped.lower().rstrip(':')
            if low in headings:
                curr = low
                sections[curr] = ""
            else:
                sections[curr] = sections.get(curr, "") + line + "\n"
        return sections
    except Exception:
        return {"header": text}

def section_scores(sections, job_keywords):
    scores = {}
    for sec, body in sections.items():
        body_clean = clean_text(body)
        if not body_clean.strip():
            scores[sec] = 0
            continue
        words = body_clean.split()
        word_count = len(words)
        hits = 0
        for kw in job_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', body_clean):
                hits += 1
        density = hits / max(1, word_count) * 100
        length_score = min(100, (word_count / 300) * 100)
        score = 0.7 * min(100, density * 5) + 0.3 * length_score
        scores[sec] = round(score, 1)
    return scores

def generate_heatmap(job_keywords, resume_text, top_n=20):
    resume_clean = clean_text(resume_text)
    freqs = []
    for kw in job_keywords:
        pattern = r'\b' + re.escape(kw) + r'\b'
        cnt = len(re.findall(pattern, resume_clean))
        freqs.append(cnt)
    df = pd.DataFrame({"keyword": job_keywords, "count": freqs})
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6, max(2, len(df) * 0.3)))
    arr = df["count"].values.reshape(-1, 1)
    im = ax.imshow(arr, aspect='auto', interpolation='nearest')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["keyword"])
    ax.set_xticks([])
    ax.set_title("Keyword Frequency in Resume")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Occurrences")
    plt.tight_layout()
    return fig, df

def create_pdf_report(out_path, data):
    if not _HAS_FPDF:
        raise RuntimeError("fpdf not installed")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 8, "ATS Resume Report", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Overall ATS Similarity Score: {data.get('score')}%", ln=True)
    pdf.ln(4)
    pdf.cell(0, 8, "Extracted Skills:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, ", ".join(data.get("skills", [])))
    pdf.ln(3)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "Top Missing Keywords:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, ", ".join(data.get("missing_keywords", [])))
    pdf.ln(3)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "Section Scores:", ln=True)
    pdf.set_font("Arial", size=11)
    for sec, sc in data.get("section_scores", {}).items():
        pdf.cell(0, 6, f"{sec.title()}: {sc}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "Extracted Resume Text (shortened):", ln=True)
    pdf.set_font("Arial", size=10)
    txt = data.get("resume_text", "")[:1500]
    pdf.multi_cell(0, 5, txt)
    pdf.output(out_path)
    return out_path

# ------------------ AI-LIKE BULLET IMPROVER (rule-based) -----------------
ACTION_VERBS = ["Led","Managed","Developed","Implemented","Designed","Automated","Improved","Optimized","Built","Created","Analyzed","Reduced","Increased","Collaborated"]

def improve_bullet(bullet: str):
    b = bullet.strip()
    passive_patterns = [
        (r'was responsible for (.+)', r'Led \1'),
        (r'was responsible for (.+)\.', r'Led \1'),
        (r'responsible for (.+)', r'Led \1'),
        (r'worked on (.+)', r'Worked on \1'),
        (r'participated in (.+)', r'Participated in \1'),
        (r'helped (?:to )?(.+)', r'Assisted with \1'),
        (r'helped to (.+)', r'Assisted with \1'),
        (r'assisted in (.+)', r'Assisted in \1')
    ]
    for pat, repl in passive_patterns:
        new = re.sub(pat, repl, b, flags=re.I)
        if new != b:
            b = new
            break
    if len(b) > 120:
        parts = re.split(r';| and |, and ', b)
        parts = [p.strip() for p in parts if len(p.strip()) > 10]
        if parts:
            b = parts[0]
    first_word = b.split()[0] if b.split() else ""
    if first_word.lower() in ("the","a","an","in","on","for","with","and"):
        b = ACTION_VERBS[0] + " " + b
    b = b[0].upper() + b[1:] if b else b
    b = b.rstrip('.')
    return b

# ------------------ STREAMLIT UI (PREMIUM DESIGN) ------------------------------
st.set_page_config(page_title="AI ATS Pro", layout="wide")
# Custom CSS for premium, glass, gradient, animations
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f1724 0%, #111827 100%);
    color: #e6eef8;
}

/* Glass Cards */
.card {
    padding: 22px;
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
    box-shadow: 0 8px 30px rgba(2,6,23,0.6);
    backdrop-filter: blur(8px);
    transition: 0.35s;
    border: 1px solid rgba(255,255,255,0.04);
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 18px 50px rgba(2,6,23,0.7);
}

/* Gradient Heading */
.title {
    font-size: 38px;
    font-weight: 800;
    background: linear-gradient(90deg, #7f5af0, #2cb67d);
    -webkit-background-clip: text;
    color: transparent;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}

/* Beautiful Button */
.stButton>button {
    background: linear-gradient(90deg, #7f5af0, #2cb67d);
    color: white;
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 15px;
    border: none;
    transition: 0.25s;
}
.stButton>button:hover {
    transform: scale(1.03);
}

/* Sidebar Design */
.block-container .sidebar .sidebar-content {
    background: linear-gradient(180deg,#0b1220,#0f1724) !important;
    border-radius: 10px;
}

/* Text Area */
textarea {
    border-radius: 10px !important;
}

/* Metric small */
.metric-card {
    padding: 14px;
    background: rgba(255,255,255,0.02);
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
}

/* small helper */
.small-muted { color: #9aa7c7; font-size: 13px; }

</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("ATS Controls")
theme_toggle = st.sidebar.checkbox("Enable Light-ish Mode", value=False)
if theme_toggle:
    st.markdown("""<style>
    .stApp { background: linear-gradient(180deg,#f6f9fc,#ffffff); color: #0f1724;}
    .stButton>button{background:linear-gradient(90deg,#06beb6,#48b1bf); color:white;}
    </style>""", unsafe_allow_html=True)

st.sidebar.markdown("## Options")
top_k = st.sidebar.slider("Top keywords to extract (job)", 5, 40, 20)
skill_fuzzy_cutoff = st.sidebar.slider("Skill fuzzy cutoff", 60, 100, 80) / 100.0
download_pdf = st.sidebar.checkbox("Enable PDF report (fpdf required)", value=True)

# Page header
st.markdown('<div class="title">🚀 AI-Powered ATS Resume Analyzer</div>', unsafe_allow_html=True)
st.write("Intelligent ATS evaluator — heatmaps, section scoring, skills extraction, missing keywords, PDF export, and bullet improver.")

# Inputs area
left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📄 Upload Resume")
    resume_file = st.file_uploader("Upload (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
    st.markdown('<div class="small-muted">Supported: PDF, DOCX, TXT. For large PDFs extraction quality may vary.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Job Description")
    job_description = st.text_area("Paste the Job Description here", height=260)
    st.markdown('<div class="small-muted">Tip: paste the full job ad (responsibilities + technical requirements) for best matching.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Custom Skills Expand
with st.expander("➕ Add Custom Skills (optional)"):
    custom_skills_raw = st.text_area("Enter keywords separated by commas", height=80)
    if custom_skills_raw:
        for s in [x.strip().lower() for x in custom_skills_raw.split(",") if x.strip()]:
            COMMON_SKILLS.add(s)

analyze_button = st.button("🔍 Analyze Resume")

# Main processing when user clicks
if analyze_button:
    if resume_file is None or not job_description.strip():
        st.error("Please upload a resume and paste the job description.")
    else:
        with st.spinner("Extracting text from resume..."):
            resume_text = extract_resume_text(resume_file)
            resume_text_clean = clean_text(resume_text)
            job_text_clean = clean_text(job_description)

        with st.spinner("Computing similarity, keywords, and extracting skills..."):
            score = calculate_similarity(resume_text_clean, job_text_clean)

            # Job keywords (top tf-idf)
            job_keywords = top_n_tfidf_terms(job_text_clean, n=top_k)
            job_keywords = [k.lower() for k in job_keywords]

            # Skills extraction from resume
            skills_found = extract_skills(resume_text)
            # fuzzy-match any job keywords to common skills (to enrich)
            for kw in job_keywords:
                fm = fuzzy_match_skill(kw, COMMON_SKILLS, cutoff=skill_fuzzy_cutoff)
                if fm:
                    if fm not in skills_found:
                        skills_found.append(fm)

            # top missing keywords (job keywords not found exactly in resume)
            missing = []
            resume_lower = resume_text_clean.lower()
            for kw in job_keywords:
                if not re.search(r'\b' + re.escape(kw) + r'\b', resume_lower):
                    missing.append(kw)
            top_missing = missing[:10]

            # sections and scores
            sections = parse_sections(resume_text)
            sec_scores = section_scores(sections, job_keywords)

        # ---------- Layout results ----------
        st.markdown("---")
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("📊 ATS Similarity Score")
            st.markdown(f'<div class="metric-card"><h2 style="margin:6px 0;">{score}%</h2><div class="small-muted">Resume ↔ Job Description match</div></div>', unsafe_allow_html=True)
            st.progress(min(100, max(0, int(score))))

            st.subheader("🧭 Top Job Keywords")
            if job_keywords:
                st.write(", ".join(job_keywords[:40]))
            else:
                st.write("_No clear keywords found_")

            st.subheader("🔎 Top Missing Keywords")
            if top_missing:
                for k in top_missing:
                    st.markdown(f"- **{k}**")
            else:
                st.write("None — your resume contains most job keywords.")

            st.subheader("🛠️ Extracted Skills")
            if skills_found:
                st.write(", ".join(sorted(set(skills_found))[:80]))
            else:
                st.write("_No skills detected with the current matcher_")

            # download report
            st.markdown("### 📥 Download Report")
            report_data = {
                "score": score,
                "skills": sorted(set(skills_found)),
                "missing_keywords": top_missing,
                "section_scores": sec_scores,
                "resume_text": resume_text[:5000]
            }

            if download_pdf and _HAS_FPDF:
                tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                try:
                    create_pdf_report(tmp_pdf.name, report_data)
                    with open(tmp_pdf.name, "rb") as f:
                        b = f.read()
                    b64 = base64.b64encode(b).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="ats_report.pdf">Download PDF report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not create PDF: {e}")
                finally:
                    try:
                        tmp_pdf.close()
                        os.unlink(tmp_pdf.name)
                    except Exception:
                        pass
            else:
                txt = []
                txt.append(f"ATS Similarity Score: {score}%\n")
                txt.append("Extracted Skills:\n" + ", ".join(sorted(set(skills_found))) + "\n")
                txt.append("Missing Keywords:\n" + ", ".join(top_missing) + "\n")
                txt.append("Section Scores:\n")
                for s, sc in sec_scores.items():
                    txt.append(f" - {s}: {sc}\n")
                txt_content = "\n".join(txt)
                b64 = base64.b64encode(txt_content.encode()).decode()
                st.markdown(f'<a href="data:file/txt;base64,{b64}" download="ats_report.txt">Download TXT report</a>', unsafe_allow_html=True)

        with col_right:
            st.subheader("📈 Keyword Matching Heatmap")
            if job_keywords:
                fig, kw_df = generate_heatmap(job_keywords[:top_k], resume_text)
                st.pyplot(fig)
            else:
                st.write("_No keywords to show heatmap_")

            st.subheader("📑 Section Quality Scores")
            if sec_scores:
                sec_df = pd.DataFrame(list(sec_scores.items()), columns=["section","score"]).sort_values("score", ascending=False)
                st.table(sec_df)
            else:
                st.write("_No sections parsed_")

        st.markdown("---")
        st.subheader("✍️ AI-like Bullet Point Improver (rule-based suggestions)")
        with st.expander("Paste bullets or lines (one per line) to get improved versions"):
            bullet_input = st.text_area("Bullet lines", height=200, key="bullet_input_area")
            if st.button("Improve Bullets", key="improve_bullets_btn"):
                if not bullet_input.strip():
                    st.info("Paste bullets to improve.")
                else:
                    bullets = [b.strip() for b in bullet_input.splitlines() if b.strip()]
                    improved = [improve_bullet(b) for b in bullets]
                    for orig, imp in zip(bullets, improved):
                        st.markdown(f"**Original:** {orig}")
                        st.markdown(f"**Suggestion:** {imp}")
                        st.markdown("---")

        st.markdown("### 📄 View Extracted Resume Text")
        with st.expander("Extracted resume (raw)"):
            st.text_area("Resume Text", resume_text, height=300)

        st.success("Analysis complete ✅")

# If not analyzed yet, show small landing/filler
if not analyze_button:
    st.markdown("---")
    st.markdown("""
    **How to use**
    1. Upload your resume (PDF/DOCX/TXT).  
    2. Paste the target job description.  
    3. Click **Analyze Resume** to get similarity, missing keywords, skills, section scoring, heatmap, and downloadable report.
    """)
    st.markdown("<div class='small-muted'>Tip: add your important custom skills in the sidebar or the 'Add Custom Skills' section for better matching.</div>", unsafe_allow_html=True)
