"""
BGU Manager - Student Portal
Similar to bgumanager.com - Flask + ChromaDB + Gemini
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import json
import warnings
import secrets
import chromadb
import google.generativeai as genai

warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"

# ==================== Config ====================
BASE_PATH = r"C:\Users\doron\PycharmProjects\PythonProject3\.venv"
GOOGLE_API_KEY = "AIzaSyDQPbz2phAK4YorWWJvwuwqgWrvM0G0t4o"
SERVER_TOKEN = secrets.token_hex(16)  # מתחדש עם כל הפעלת שרת – מאפס credentials
BAD_VALS = {'nan','none','no data','error','label not found','no grade found',
            "no 'average' found in text","no 'average' found",'empty pdf / image',
            'server error','','n/a'}

# ==================== Data Sources (with labels) ====================
DATA_SOURCES = {
    "grades": {
        "label": "ציונים להנדסת חשמל ומחשבים",
        "file": os.path.join(BASE_PATH, "grades.csv"),
        "icon": "📊",
        "desc": "ממוצעי ציונים בקורסי הנדסת חשמל ומחשבים",
        "keywords": ["ציון", "ממוצע", "קורס", "סמסטר", "חשמל", "מחשבים", "2025", "2024", "2023"],
    },
    "machinery2": {
        "label": "ציונים להנדסת מכונות",
        "file": os.path.join(BASE_PATH, "downloads", "machinery2_with_grades.csv"),
        "icon": "🔧",
        "desc": "ממוצעי ציונים בקורסי הנדסת מכונות",
        "keywords": ["מכונות", "מכונה", "מכונאות", "machinery"],
    },
    "all_courses": {
        "label": "ציונים",
        "file": os.path.join(BASE_PATH, "downloads", "g_all_courses_with_grades.csv"),
        "icon": "📈",
        "desc": "ציונים ממוצעים לכל הקורסים באוניברסיטה",
        "keywords": ["קורסים", "כל הקורסים", "נקודות זכות", 'נק"ז'],
    },
    "admission": {
        "label": "תנאי קבלה",
        "file": os.path.join(BASE_PATH, "bgu_admission.csv"),
        "icon": "📋",
        "desc": "תנאי קבלה לכל המחלקות",
        "keywords": ["קבלה", "תנאי", "כניסה", "פסיכומטרי", "בגרות", "סכם", "מינימום", "להתקבל"],
    },
    "projects": {
        "label": "פרויקט גמר להנדסת חשמל",
        "file": os.path.join(BASE_PATH, "Projects_Classified.csv"),
        "icon": "🔬",
        "desc": "פרויקטי גמר בהנדסת חשמל ומחשבים",
        "keywords": ["פרויקט", "גמר", "מנחה", "פרויקטים", "adviser", "specialization"],
    },
    "scholarships": {
        "label": "מלגות",
        "file": os.path.join(BASE_PATH, "bgu_scholarships_new_3.csv"),
        "icon": "🎓",
        "desc": "מלגות לסטודנטים בבן גוריון",
        "keywords": ["מלגה", "מלגות", "זכאות", "מילואים", "חרבות ברזל"],
    },
    "army": {
        "label": "מילואים וזכויות",
        "file": os.path.join(BASE_PATH, "army_results.json"),
        "icon": "🪖",
        "desc": "זכויות סטודנטים משרתי מילואים",
        "keywords": ["מילואים", "צבא", "חרבות ברזל", "זכויות", "שירות"],
    },
    "people": {
        "label": "אנשי סגל",
        "file": os.path.join(BASE_PATH, "..", "data", "people_of_bgu.csv"),
        "icon": "👥",
        "desc": "אנשי סגל, חוקרים ועובדים באוניברסיטת בן גוריון",
        "keywords": ["סגל", "מרצה", "פרופסור", "דוקטור", "חוקר", "מנחה", "עובד", "staff"],
    },
    "partner_knowledge": {
        "label": "מידע כללי BGU",
        "file": os.path.join(BASE_PATH, "downloads", "bgu_partner_knowledge.csv"),
        "icon": "🏫",
        "desc": "מידע כללי על האוניברסיטה – מזכירות, הרשמה, תוכניות, שירותים",
        "keywords": ["מזכירות", "הרשמה", "תוכנית", "שירות", "פקולטה", "מחלקה", "לוח שנה",
                     "תקנון", "נוהל", "בחינה", "אתר", "קשר", "אוניברסיטה", "כללי"],
    },
}

NOT_FOUND_PHRASES = [
    "לא מצאתי", "לא נמצא", "אין מידע", "לא קיים", "לא נמצאו",
    "אין לי מידע", "לא עולה", "לא מופיע", "לא ברשותי", "לא כלול",
    "לא זמין", "לא מצוי", "אין נתון",
]

def detect_sources(question: str) -> list:
    """מזהה מילות מפתח ומחזיר רשימת מקורות רלוונטיים."""
    q = question.lower()
    matched = [key for key, src in DATA_SOURCES.items()
               if any(kw.lower() in q for kw in src["keywords"])]
    return matched if matched else list(DATA_SOURCES.keys())

app = Flask(__name__,
            template_folder=os.path.join(BASE_PATH, "templates"),
            static_folder=os.path.join(BASE_PATH, "static"))

# ==================== Data Loading ====================
def safe_csv(path):
    if not os.path.exists(path):
        print(f"  [MISSING] {os.path.basename(path)}")
        return pd.DataFrame()
    for enc in ['utf-8', 'utf-8-sig', 'windows-1255']:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"  [OK] {os.path.basename(path)}: {len(df)} rows")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(path)}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def load_all_grades():
    """
    ממזג את כל קבצי הציונים:
    - grades.csv
    - g_all_courses_with_grades.csv
    - machinery2_with_grades.csv
    - management_grades_extracted.csv
    מנרמל שמות עמודות, מסיר כפילויות לפי מספר קורס,
    ממיין לפי מספר קורס ואז עמודות מהשנה הכי חדשה לישנה.
    """
    import re

    GRADE_FILES = [
        (os.path.join(BASE_PATH, "grades.csv"), None),
        (os.path.join(BASE_PATH, "downloads", "g_all_courses_with_grades.csv"), None),
        (os.path.join(BASE_PATH, "downloads", "machinery2_with_grades.csv"), None),
        (os.path.join(BASE_PATH, "management_grades_extracted.csv"), {
            'Course_Number': 'מספר קורס',
            'Course_Name':   'שם קורס',
            'Prerequisites': 'קורסי חובת מעבר',
        }),
    ]

    def norm_sem(col):
        """'2025 A' -> '2025A'"""
        return re.sub(r'(\d{4})\s+([AB])', r'\1\2', str(col).strip())

    def is_grade_col(col):
        return bool(re.match(r'\d{4}[AB]', norm_sem(col)))

    def sem_sort_key(col):
        m = re.match(r'(\d{4})([AB])', col)
        if m:
            return (-int(m.group(1)), 0 if m.group(2) == 'B' else 1)
        return (0, 0)

    FIXED = ['מספר קורס', 'שם קורס', 'נק"ז', 'קורסי חובת מעבר']

    all_grade_cols = set()
    dfs_loaded = []

    for path, rename_map in GRADE_FILES:
        df = safe_csv(path)
        if df.empty:
            continue
        if rename_map:
            df = df.rename(columns=rename_map)
        # Normalize semester column names
        df = df.rename(columns={c: norm_sem(c) for c in df.columns})
        if 'מספר קורס' not in df.columns:
            continue
        df['מספר קורס'] = df['מספר קורס'].astype(str).str.strip()
        for col in df.columns:
            if is_grade_col(col):
                all_grade_cols.add(col)
        dfs_loaded.append(df)

    if not dfs_loaded:
        return pd.DataFrame(), []

    sorted_grade_cols = sorted(all_grade_cols, key=sem_sort_key)

    # Merge: course_num -> dict
    merged = {}
    for df in dfs_loaded:
        df = df.fillna('')
        for _, row in df.iterrows():
            cnum = str(row.get('מספר קורס', '')).strip()
            # Skip invalid rows
            if not cnum or not re.match(r'^\d{5,}', cnum):
                continue
            if cnum not in merged:
                merged[cnum] = {'מספר קורס': cnum,
                                'שם קורס': '', 'נק"ז': '', 'קורסי חובת מעבר': ''}
            rec = merged[cnum]
            # Fill text fields if empty
            for col in ['שם קורס', 'נק"ז', 'קורסי חובת מעבר']:
                val = str(row.get(col, '')).strip()
                if val and val.lower() not in BAD_VALS and not rec.get(col):
                    rec[col] = val
            # Fill grade fields (prefer existing non-empty)
            for col in sorted_grade_cols:
                val = str(row.get(col, '')).strip()
                if val and val.lower() not in BAD_VALS and not rec.get(col):
                    rec[col] = val

    if not merged:
        return pd.DataFrame(), []

    result_df = pd.DataFrame(list(merged.values()))

    # Sort by course number numerically
    def num_key(n):
        try: return int(re.sub(r'\D', '', str(n)))
        except: return 999999999

    result_df['_sort'] = result_df['מספר קורס'].apply(num_key)
    result_df = result_df.sort_values('_sort').drop('_sort', axis=1).reset_index(drop=True)

    # Column order: fixed + grade cols (newest first) + rest
    final_cols = [c for c in FIXED if c in result_df.columns]
    final_cols += [c for c in sorted_grade_cols if c in result_df.columns]
    final_cols += [c for c in result_df.columns if c not in final_cols]
    result_df = result_df[final_cols].fillna('')

    print(f"  [MERGED] Courses total: {len(result_df)} | Grade cols: {sorted_grade_cols}")
    return result_df, sorted_grade_cols


print("\nLoading CSV data...")
courses_df, GRADE_COLS = load_all_grades()
admission_df         = safe_csv(os.path.join(BASE_PATH, "bgu_admission.csv"))
admission_complete_df= safe_csv(os.path.join(BASE_PATH, "bgu_admission_complete.csv"))
projects_df          = safe_csv(os.path.join(BASE_PATH, "Projects_Classified.csv"))
scholarships_df      = safe_csv(os.path.join(BASE_PATH, "bgu_scholarships_new_3.csv"))

# קבצים נפרדים לשימוש בצ'אט (נשלחים במלואם ל-Gemini כמו AskUni)
grades_df       = safe_csv(os.path.join(BASE_PATH, "grades.csv"))
machinery2_df   = safe_csv(os.path.join(BASE_PATH, "machinery2.csv"))
all_courses_df  = safe_csv(os.path.join(BASE_PATH, "downloads", "g_all_courses_with_grades.csv"))

# ==================== Army / Reserves Data ====================
import json as _army_json

def load_army_data():
    path = os.path.join(BASE_PATH, "army_results.json")
    if not os.path.exists(path):
        print("  [INFO] army_results.json not found – run army.py first")
        return pd.DataFrame()
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = _army_json.load(f)
        df = pd.DataFrame(records)
        print(f"  [OK] army_results.json: {len(df)} records")
        return df
    except Exception as e:
        print(f"  [WARN] army_results.json load error: {e}")
        return pd.DataFrame()

army_df = load_army_data()

people_df = safe_csv(os.path.join(BASE_PATH, "..", "data", "people_of_bgu.csv"))
partner_knowledge_df = safe_csv(os.path.join(BASE_PATH, "downloads", "bgu_partner_knowledge.csv"))

# מיפוי מקורות -> DataFrames (לשימוש בצ'אט עם שליחת CSV מלא כמו AskUni)
SOURCE_DFS: dict = {}

def _refresh_source_dfs():
    SOURCE_DFS.update({
        "grades":      grades_df,
        "machinery2":  machinery2_df,
        "all_courses": all_courses_df,
        "admission":   admission_complete_df if not admission_complete_df.empty else admission_df,
        "projects":    projects_df,
        "scholarships":scholarships_df,
        "army":        army_df,
        "people":      people_df,
        "partner_knowledge": partner_knowledge_df,
    })

_refresh_source_dfs()


def build_csv_context(source_keys: list) -> str:
    """שולח CSV מלא של כל מקור רלוונטי – זהה ל-AskUni.py."""
    parts = []
    for key in source_keys:
        df = SOURCE_DFS.get(key)
        if df is None or (hasattr(df, 'empty') and df.empty):
            continue
        src = DATA_SOURCES[key]
        parts.append(
            f"\n--- {src['icon']} {src['label']} ({src['desc']}) ---\n"
            + df.to_csv(index=False)
        )
    return "\n".join(parts) if parts else "אין נתונים זמינים."


# ==================== RAG עם דירוג ====================
_RAG_N_RESULTS = 8    # כמה מסמכים לשלוף מ-ChromaDB
_RAG_MIN_SCORE = 0.15 # סף מינימום רלוונטיות (1 - distance)


def build_rag_context(query: str) -> str:
    """
    RAG עם דירוג – שולף את X המסמכים הרלוונטיים ביותר מ-ChromaDB.
    ממיין לפי ציון רלוונטיות (גבוה = רלוונטי יותר).
    """
    if not collection:
        # fallback: keyword search על כל הקבצים
        return build_csv_context(list(DATA_SOURCES.keys()), query=query)

    try:
        res = collection.query(query_texts=[query], n_results=_RAG_N_RESULTS)
        docs  = (res.get("documents")  or [[]])[0]
        dists = (res.get("distances")  or [[]])[0]
        metas = (res.get("metadatas")  or [[]])[0]

        if not docs:
            return "לא נמצאו תוצאות."

        # מיון: הכי קרוב (distance קטן) = הכי רלוונטי
        ranked = sorted(zip(docs, dists, metas), key=lambda x: x[1])

        parts = []
        for doc, dist, meta in ranked:
            score = round(1 - dist, 3)
            if score < _RAG_MIN_SCORE:
                continue
            doc_type = meta.get("type", "")
            source   = meta.get("source", "")
            parts.append(f"[רלוונטיות: {score}] [{doc_type}] [{source}]\n{doc}")

        if not parts:
            return "לא נמצאו תוצאות רלוונטיות."

        result = "\n\n---\n".join(parts)
        print(f"[RAG] {len(parts)} מסמכים, {len(result)} תווים")
        return result

    except Exception as e:
        print(f"[RAG error] {e}")
        return build_csv_context(list(DATA_SOURCES.keys()), query=query)


# ==================== Gemini + ChromaDB ====================
genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel('gemini-2.5-flash')

# ChromaDB – נבנה ע"י ALL_CLAUDE.py
_CHROMA_PATH = os.path.join(BASE_PATH, "..", "chroma_db_storage")
_COLLECTION_NAME = "bgu_knowledge"

collection = None
try:
    chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)
    collection = chroma_client.get_collection(_COLLECTION_NAME)
    print(f"  [OK] ChromaDB (ALL_CLAUDE): {collection.count()} records")
except Exception as e:
    print(f"  [WARN] ChromaDB לא נמצא – הרץ ALL_CLAUDE.py תחילה: {e}")

# ==================== Helpers ====================
def clean(v):
    s = str(v).strip()
    return '' if s.lower() in BAD_VALS else s

def get_latest_grade(row):
    for col in reversed(GRADE_COLS):
        v = clean(str(row.get(col, '')))
        if v:
            return v, col
    return '', ''

def row_to_dict(row):
    return {k: clean(v) for k, v in row.items()}

# ==================== Routes ====================
LIKES_FILE = os.path.join(BASE_PATH, 'likes.json')

def _read_likes() -> int:
    try:
        with open(LIKES_FILE, 'r') as f:
            return json.load(f).get('likes', 0)
    except Exception:
        return 0

def _save_likes(n: int) -> None:
    with open(LIKES_FILE, 'w') as f:
        json.dump({'likes': n}, f)

@app.route('/api/like', methods=['GET', 'POST'])
def api_like():
    if request.method == 'POST':
        n = _read_likes() + 1
        _save_likes(n)
        return jsonify({'likes': n})
    return jsonify({'likes': _read_likes()})


@app.route('/')
def index():
    stats = {
        'courses':      len(courses_df),
        'admission':    len(admission_df),
        'projects':     len(projects_df),
        'scholarships': len(scholarships_df),
        'ai_records':   collection.count() if collection else 0,
        'grade_cols':   len(GRADE_COLS),
    }
    return render_template('bgu_manager.html', stats=stats, server_token=SERVER_TOKEN)

# ---------- COURSES ----------
@app.route('/api/courses')
def api_courses():
    q       = request.args.get('q', '').strip().lower()
    num_q   = request.args.get('num', '').strip()
    page    = max(1, int(request.args.get('page', 1)))
    per_page = 20

    if courses_df.empty:
        return jsonify({'courses': [], 'total': 0})

    df = courses_df.fillna('')
    name_col = 'שם קורס' if 'שם קורס' in df.columns else df.columns[0]
    num_col  = 'מספר קורס' if 'מספר קורס' in df.columns else None

    if q:
        mask = df[name_col].astype(str).str.lower().str.contains(q, na=False)
        if num_col:
            mask |= df[num_col].astype(str).str.lower().str.contains(q, na=False)
        df = df[mask]
    if num_q and num_col:
        df = df[df[num_col].astype(str).str.contains(num_q, na=False)]

    total = len(df)
    chunk = df.iloc[(page-1)*per_page : page*per_page]

    results = []
    for _, row in chunk.iterrows():
        grade, sem = get_latest_grade(row)
        # Build grades dict: col -> value (only non-empty)
        grades = {col: clean(str(row.get(col,''))) for col in GRADE_COLS if clean(str(row.get(col,'')))}
        results.append({
            'id':       clean(str(row.get('מספר קורס', ''))),
            'name':     clean(str(row.get('שם קורס', ''))),
            'credits':  clean(str(row.get('נק"ז', ''))),
            'grade':    grade,
            'semester': sem,
            'grades':   grades,
        })
    return jsonify({'courses': results, 'total': total, 'page': page,
                    'per_page': per_page, 'grade_cols': GRADE_COLS})

@app.route('/api/courses/<course_id>')
def api_course_detail(course_id):
    if courses_df.empty:
        return jsonify({'error': 'No data'}), 404
    df = courses_df.fillna('')
    if 'מספר קורס' not in df.columns:
        return jsonify({'error': 'No course ID column'}), 404
    row_df = df[df['מספר קורס'].astype(str) == course_id]
    if row_df.empty:
        return jsonify({'error': 'Not found'}), 404
    row = row_df.iloc[0]
    history = {col: clean(str(row.get(col,''))) for col in GRADE_COLS if clean(str(row.get(col,'')))}
    return jsonify({
        'id':      clean(str(row.get('מספר קורס',''))),
        'name':    clean(str(row.get('שם קורס',''))),
        'credits': clean(str(row.get('נק"ז',''))),
        'prereqs': clean(str(row.get('קורסי חובת מעבר',''))),
        'history': history,
    })

# ---------- ADMISSION ----------
@app.route('/api/admission')
def api_admission():
    q = request.args.get('q', '').strip().lower()
    if admission_df.empty:
        return jsonify({'tracks': [], 'total': 0})
    df = admission_df.fillna('')
    name_col = 'שם_המסלול' if 'שם_המסלול' in df.columns else df.columns[0]
    if q:
        df = df[df[name_col].astype(str).str.lower().str.contains(q, na=False)]
    results = []
    for _, row in df.iterrows():
        t = {
            'name':    clean(str(row.get('שם_המסלול',''))),
            'url':     clean(str(row.get('URL',''))),
            'type':    clean(str(row.get('סוג_קבלה',''))),
            'score':   clean(str(row.get('סכם',''))),
            'psycho':  clean(str(row.get('פסיכומטרי',''))),
            'bagrut':  clean(str(row.get('ממוצע_בגרות',''))),
            'math':    clean(str(row.get('מתמטיקה',''))),
            'english': clean(str(row.get('אנגלית',''))),
            'physics': clean(str(row.get('פיסיקה',''))),
            'eng_score': clean(str(row.get('סכם_הנדסה',''))),
        }
        if t['name']:
            results.append(t)
    return jsonify({'tracks': results, 'total': len(results)})

def _proj_display_year(pid: str) -> str:
    """גוזר שנת תצוגה מ-Project ID: p-2026-* → 2026, p-2025-* → 2025, אחר → 2000."""
    s = str(pid).strip()
    if s.startswith('p-2026'):
        return '2026'
    if s.startswith('p-2025'):
        return '2025'
    return '2000'


# ---------- PROJECTS ----------
@app.route('/api/reports/<folder>')
def api_reports(folder):
    allowed = {'PDR', 'preliminary', 'progress'}
    if folder not in allowed:
        return jsonify({'files': []})
    path = os.path.join(app.static_folder, 'reports', folder)
    os.makedirs(path, exist_ok=True)
    files = sorted(f for f in os.listdir(path) if f.lower().endswith('.pdf'))
    return jsonify({'files': files})


@app.route('/api/projects')
def api_projects():
    q        = request.args.get('q', '').strip().lower()
    track    = request.args.get('track', '').strip()
    year     = request.args.get('year', '').strip()
    page     = max(1, int(request.args.get('page', 1)))
    per_page = 20

    if projects_df.empty:
        return jsonify({'projects': [], 'total': 0, 'tracks': [], 'years': []})

    df = projects_df.fillna('').copy()
    name_col       = next((c for c in df.columns if 'Project Name' in c or 'Name' in c), df.columns[0])
    spec_col       = next((c for c in df.columns if 'Specialization' in c or 'תחום' in c), None)
    supervisor_col = next((c for c in df.columns if 'Supervisor' in c or 'Adviser' in c or 'מנחה' in c), None)
    pid_col        = next((c for c in df.columns if 'Project ID' in c), None)

    # שנת תצוגה לפי Project ID
    df['_display_year'] = df[pid_col].apply(_proj_display_year) if pid_col else '2000'

    if q:
        mask = df[name_col].astype(str).str.lower().str.contains(q, na=False)
        if supervisor_col:
            mask |= df[supervisor_col].astype(str).str.lower().str.contains(q, na=False)
        df = df[mask]
    if track and spec_col:
        df = df[df[spec_col].astype(str).str.contains(track, case=False, na=False)]
    if year:
        df = df[df['_display_year'] == year]

    total = len(df)
    chunk = df.iloc[(page-1)*per_page : page*per_page]

    results = [{col: clean(str(row[col])) for col in df.columns} for _, row in chunk.iterrows()]

    all_tracks = []
    if spec_col:
        all_tracks = sorted(set(v for v in projects_df[spec_col].dropna().unique() if v))

    return jsonify({'projects': results, 'total': total, 'tracks': all_tracks,
                    'years': ['2026', '2025', '2000'],
                    'cols': {'name': name_col, 'spec': spec_col,
                             'supervisor': supervisor_col}})

# ---------- SCHOLARSHIPS ----------
import json as _json

def parse_schol_row(idx, row):
    """Parse a scholarships row into a structured dict."""
    raw_json = str(row.get('תוכן_נקי_JSON', '') or '')
    parsed = {}
    try:
        parsed = _json.loads(raw_json) if raw_json.strip().startswith('{') else {}
    except Exception:
        pass
    return {
        '_idx': idx,
        'name': clean(str(row.get('שם המלגה', '') or parsed.get('שם_המלגה', ''))),
        'url':  clean(str(row.get('URL', ''))),
        'desc':     parsed.get('תיאור_כללי', ''),
        'target':   parsed.get('קהל_יעד', ''),
        'eligibility': parsed.get('תנאי_זכאות', ''),
        'amount':   parsed.get('סכום_המלגה', ''),
        'dates':    parsed.get('תאריכי_הרשמה', ''),
        'obligations': parsed.get('התחייבויות', ''),
        'extra':    {k: v for k, v in parsed.items()
                     if k not in ('שם_המלגה','תיאור_כללי','קהל_יעד','תנאי_זכאות',
                                  'סכום_המלגה','תאריכי_הרשמה','התחייבויות')},
    }

@app.route('/api/scholarships')
def api_scholarships():
    q = request.args.get('q', '').strip().lower()
    if scholarships_df.empty:
        return jsonify({'scholarships': [], 'total': 0})
    df = scholarships_df.fillna('')

    # Search in all text (including JSON content)
    if q:
        mask = pd.Series([False]*len(df), index=df.index)
        for col in df.columns:
            mask |= df[col].astype(str).str.lower().str.contains(q, na=False, regex=False)
        df = df[mask]

    records = [parse_schol_row(i, row) for i, (_, row) in enumerate(df.head(100).iterrows())]
    return jsonify({'scholarships': records, 'total': len(df)})


@app.route('/api/scholarships/ask', methods=['POST'])
def api_scholarships_ask():
    """AI-powered scholarship search: describe what you need, get matched scholarships."""
    data = request.get_json()
    query = (data.get('query') or '').strip()
    if not query:
        return jsonify({'error': 'שאלה ריקה'}), 400
    if scholarships_df.empty:
        return jsonify({'error': 'אין נתוני מלגות'}), 500

    # Build full context from all scholarships
    all_schols = []
    for i, (_, row) in enumerate(scholarships_df.fillna('').iterrows()):
        s = parse_schol_row(i, row)
        parts = [f"מלגה #{i+1}: {s['name']}"]
        if s.get('url'):
            parts.append(f"  URL: {s['url']}")
        for field, label in [('desc','תיאור'), ('target','קהל יעד'),
                              ('eligibility','תנאי זכאות'), ('amount','סכום'),
                              ('dates','תאריכים'), ('obligations','התחייבויות')]:
            if s.get(field):
                parts.append(f"  {label}: {s[field][:300]}")
        all_schols.append("\n".join(parts))

    schols_text = "\n\n".join(all_schols)

    prompt = f"""אתה יועץ אקדמי מומחה לסטודנטים באוניברסיטת בן גוריון.
להלן רשימת כל המלגות הזמינות עם הפרטים שלהן:

{schols_text}

---
שאלת הסטודנט: {query}

המשימה שלך:
1. נתח את הצרכים של הסטודנט
2. זהה את המלגות הכי מתאימות מתוך הרשימה לעיל
3. ענה בעברית בצורה ברורה ומסודרת:
   - פרט אילו מלגות מתאימות ולמה
   - ציין את מספר המלגה (# מספר) בסוגריים בסוף שם כל מלגה
   - הסבר בקצרה מדוע כל מלגה מתאימה לפרופיל שתואר
   - ציין אם יש דרישות מיוחדות שצריך לשים לב אליהן
   - אם למלגה יש URL – הוסף: 🔗 [לחץ כאן להגשה](URL)
4. אם אין מלגה מתאימה – ציין זאת בכנות"""

    try:
        resp = gemini.generate_content(prompt)
        answer = resp.text

        # Extract scholarship numbers mentioned in the response to return matched cards
        import re
        mentioned_nums = [int(m)-1 for m in re.findall(r'#(\d+)', answer)
                          if 0 <= int(m)-1 < len(scholarships_df)]
        mentioned_names = re.findall(r'מלגת?\s+[\w\s״׳"\']+', answer)

        matched = []
        seen_idx = set()
        for idx in mentioned_nums:
            if idx not in seen_idx:
                seen_idx.add(idx)
                row = scholarships_df.fillna('').iloc[idx]
                matched.append(parse_schol_row(idx, row))

        return jsonify({'answer': answer, 'matched': matched})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------- PEOPLE / STAFF ----------
@app.route('/api/people')
def api_people():
    q        = request.args.get('q', '').strip().lower()
    page     = max(1, int(request.args.get('page', 1)))
    per_page = 30

    if people_df.empty:
        return jsonify({'people': [], 'total': 0})

    df = people_df.fillna('')

    if q:
        mask = (
            df['name'].astype(str).str.lower().str.contains(q, na=False) |
            df['email'].astype(str).str.lower().str.contains(q, na=False) |
            df['faculty'].astype(str).str.lower().str.contains(q, na=False)
        )
        df = df[mask]

    total = len(df)
    chunk = df.iloc[(page - 1) * per_page: page * per_page]
    records = chunk.fillna('').to_dict(orient='records')
    return jsonify({'people': records, 'total': total, 'page': page, 'per_page': per_page})


# ---------- ARMY / RESERVES ----------
@app.route('/api/army/results')
def api_army_results():
    if army_df.empty:
        return jsonify({'results': [], 'total': 0})
    records = army_df.fillna('').to_dict(orient='records')
    return jsonify({'results': records, 'total': len(records)})


@app.route('/api/army/ask', methods=['POST'])
def api_army_ask():
    data = request.get_json()
    msg  = (data.get('message') or '').strip()
    if not msg:
        return jsonify({'error': 'שאלה ריקה'}), 400

    # Build context from army data
    context = ''
    if not army_df.empty:
        relevant = []
        keywords = [w for w in msg.lower().split() if len(w) > 1]
        for _, row in army_df.fillna('').iterrows():
            text = str(row.get('text', '')) + str(row.get('title', ''))
            if any(kw in text.lower() for kw in keywords):
                relevant.append(f"כותרת: {row.get('title','')}\nמקור: {row.get('url','')}\n{str(row.get('text',''))[:1500]}")
        if not relevant:
            relevant = [f"כותרת: {r.get('title','')}\n{str(r.get('text',''))[:800]}"
                        for r in army_df.fillna('').head(8).to_dict(orient='records')]
        context = "\n\n---\n\n".join(relevant[:6])
    else:
        context = "אין נתוני מילואים זמינים. נא להריץ את army.py תחילה."

    prompt = f"""אתה AskUni – עוזר אקדמי לסטודנטים של אוניברסיטת בן גוריון.
אתה מתמחה בזכויות סטודנטים משרתי מילואים.

מידע רלוונטי ממסד הנתונים:
{context}

חוקים:
- ענה בעברית בלבד, בצורה ברורה ומסודרת
- התבסס אך ורק על המידע שלמעלה
- אם המידע לא קיים – ציין זאת במפורש
- השתמש בנקודות/רשימות כשמתאים

שאלת הסטודנט: {msg}"""

    try:
        resp = gemini.generate_content(prompt)
        return jsonify({'answer': resp.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------- SETTINGS ----------

@app.route('/api/settings')
def api_settings():
    """מחזיר מידע על כל מקורות הנתונים הזמינים."""
    result = []
    df_map = {
        'grades':      courses_df,
        'admission':   admission_df,
        'projects':    projects_df,
        'scholarships': scholarships_df,
        'army':        army_df,
        'people':      people_df,
        'machinery2':  pd.DataFrame(),  # נטען בתוך courses_df הממוזג
        'all_courses': pd.DataFrame(),
        'partner_knowledge': partner_knowledge_df,
    }
    for key, src in DATA_SOURCES.items():
        df = df_map.get(key, pd.DataFrame())
        loaded = not df.empty if key not in ('machinery2', 'all_courses') else os.path.exists(src['file'])
        result.append({
            'key':    key,
            'label':  src['label'],
            'icon':   src['icon'],
            'desc':   src['desc'],
            'file':   os.path.basename(src['file']),
            'loaded': loaded,
            'rows':   len(df) if not df.empty else None,
        })
    return jsonify({
        'sources': result,
        'chroma':  collection.count() if collection else 0,
        'routing': 'smart – keyword detection + fallback to all sources',
    })


# ---------- AI CHAT ----------

def df_keyword_search(df, query, label, max_rows=30):
    """Search a DataFrame for rows matching any keyword in query. Returns (text_block, row_count)."""
    if df.empty:
        return '', 0
    keywords = [w for w in query.lower().split() if len(w) > 1]
    if not keywords:
        return '', 0
    df = df.fillna('').astype(str)
    mask = pd.Series([False] * len(df), index=df.index)
    for kw in keywords:
        for col in df.columns:
            mask |= df[col].str.lower().str.contains(kw, na=False, regex=False)
    matched = df[mask].head(max_rows)
    if matched.empty:
        return '', 0
    lines = [f"--- {label} ({len(matched)} שורות רלוונטיות) ---"]
    for _, row in matched.iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns if row[col].strip() and row[col].lower() not in BAD_VALS]
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines), len(matched)


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """
    צ'אט AI – שולח CSV מלא ל-Gemini (כמו AskUni) עם ניתוב חכם ו-fallback.
    שלב 1: מזהה מילות מפתח → שולח רק את הקבצים הרלוונטיים במלואם.
    שלב 2: אם לא נמצא → שולח את כל הקבצים.
    """
    data = request.get_json()
    msg  = (data.get('message') or '').strip()
    if not msg:
        return jsonify({'error': 'הודעה ריקה'}), 400

    sources_list = "\n".join(
        f"  {s['icon']} {s['label']} – {s['desc']}"
        for s in DATA_SOURCES.values()
    )

    def make_prompt(context: str, is_fallback: bool = False) -> str:
        fallback_note = "\n(חיפוש מורחב בכל מקורות המידע)\n" if is_fallback else ""
        return (
            "אתה AskUni – עוזר אקדמי חכם ואדיב לסטודנטים של אוניברסיטת בן גוריון בנגב.\n"
            f"מקורות המידע שלך:\n{sources_list}\n{fallback_note}\n"
            "הנחיות:\n"
            "- ענה בעברית בלבד, בצורה ברורה ומסודרת\n"
            "- ענה על סמך המידע בטבלאות בלבד\n"
            "- אם שאלו על ציון, חפש לפי שם קורס או מספרו\n"
            "- אם שאלו על תנאי קבלה, פרט את כל הדרישות\n"
            "- אם שאלו על מלגות, זהה את המתאימות ביותר לפי הפרופיל\n"
            "- אם המידע לא קיים – ציין זאת במפורש\n"
            "- השתמש בנקודות/רשימות כשמתאים\n"
            "- קישורים: אם בנתונים יש עמודת URL לאותה מלגה או מסלול קבלה – "
            "הוסף בסוף הרלוונטי: 🔗 [לחץ כאן לפרטים נוספים](URL). "
            "אם יש כמה מלגות/מסלולים, הוסף קישור לכל אחד מהם. אל תשמיט קישורים.\n\n"
            f"נתונים:\n{context}\n\n"
            f"שאלת המשתמש: {msg}"
        )

    try:
        # שלב 1: ניתוב חכם – שלח CSV מלא של מקורות ממוקדים (זהה ל-AskUni)
        target_keys = detect_sources(msg)
        is_all      = set(target_keys) == set(DATA_SOURCES.keys())

        context1 = build_csv_context(target_keys)
        resp1    = gemini.generate_content(make_prompt(context1))
        answer   = resp1.text

        # שלב 2: fallback – אם לא נמצא, שלח את כל הקבצים
        used_fallback = False
        if any(p in answer for p in NOT_FOUND_PHRASES) and not is_all:
            context2  = build_csv_context(list(DATA_SOURCES.keys()))
            resp2     = gemini.generate_content(make_prompt(context2, is_fallback=True))
            answer    = resp2.text
            used_fallback = True
            target_keys   = list(DATA_SOURCES.keys())

        return jsonify({
            'answer':    answer,
            'routed_to': [DATA_SOURCES[k]['label'] for k in target_keys if k in DATA_SOURCES],
            'fallback':  used_fallback,
            'has_data':  True,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== Timetable (Course Schedule) ====================
import re as _re

_DAY_NAMES = {'א': 'ראשון', 'ב': 'שני', 'ג': 'שלישי', 'ד': 'רביעי', 'ה': 'חמישי'}


def _parse_time_slots(times_text: str) -> list:
    """'ג 12:00-14:00' → [{day, day_name, start, end}, ...]"""
    slots = []
    if not times_text:
        return slots
    pattern = r'([אבגדה](?:[,\s]*[אבגדה])*)\s+(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})'
    for m in _re.finditer(pattern, times_text):
        days  = _re.findall(r'[אבגדה]', m.group(1))
        start = m.group(2).zfill(5)
        end   = m.group(3).zfill(5)
        for day in days:
            slots.append({'day': day, 'day_name': _DAY_NAMES.get(day, day),
                          'start': start, 'end': end})
    return slots


def _scrape_timetable(department: str, degree: str, course_num: str,
                      year: str, semester: str) -> dict:
    """Selenium visible browser – scrapes BGU course schedule (mirrors Course_file.py)."""
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select
    from selenium.webdriver.chrome.options import Options
    import time as _time
    import re as _re2

    BGU_URL  = "https://bgu4u.bgu.ac.il/pls/scwp/!app.gate?app=ann"
    BGU_FORM = "https://bgu4u.bgu.ac.il/pls/scwp/!app.ann?lang=he"

    opts = Options()
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--disable-infobars")
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])

    driver = webdriver.Chrome(options=opts)

    def _try_select(sel_el, *values):
        sel = Select(sel_el)
        for v in values:
            try: sel.select_by_value(v); return True
            except Exception: pass
            for opt in sel.options:
                if v in (opt.get_attribute("value") or "") or v in opt.text:
                    try: sel.select_by_value(opt.get_attribute("value")); return True
                    except Exception:
                        try: opt.click(); return True
                        except Exception: pass
        return False

    def _fill(field_id, value):
        for loc in [(By.ID, field_id), (By.NAME, field_id)]:
            try:
                el = driver.find_element(*loc)
                if el.tag_name.lower() == "select":
                    return _try_select(el, value)
                el.clear(); el.send_keys(value); return True
            except Exception:
                pass
        return False

    def _switch_best_frame():
        driver.switch_to.default_content()
        best_n, best_idx = len(driver.find_elements(By.TAG_NAME, "select")), None
        frames = (driver.find_elements(By.TAG_NAME, "iframe") +
                  driver.find_elements(By.TAG_NAME, "frame"))
        for idx, fr in enumerate(frames):
            try:
                driver.switch_to.frame(fr)
                n = len(driver.find_elements(By.TAG_NAME, "select"))
                driver.switch_to.default_content()
                if n > best_n:
                    best_n, best_idx = n, idx
            except Exception:
                driver.switch_to.default_content()
        if best_idx is not None:
            frames = (driver.find_elements(By.TAG_NAME, "iframe") +
                      driver.find_elements(By.TAG_NAME, "frame"))
            driver.switch_to.frame(frames[best_idx])

    def _switch_main_frame():
        driver.switch_to.default_content()
        try:
            driver.switch_to.frame("main")
        except Exception:
            pass

    def _click_advanced():
        try:
            clicked = driver.execute_script("""
                var keywords = ['מורחב','morch','adv','Advanced'];
                var tags = ['a','input','button','area','img','span','div','td'];
                for (var t=0; t<tags.length; t++) {
                    var els = document.getElementsByTagName(tags[t]);
                    for (var i=0; i<els.length; i++) {
                        var e = els[i];
                        var hay = (e.textContent||'') + (e.value||'') +
                                  (e.alt||'') + (e.title||'') +
                                  (e.href||'') + (e.src||'') + (e.name||'');
                        for (var k=0; k<keywords.length; k++) {
                            if (hay.indexOf(keywords[k]) !== -1) {
                                e.click(); return true;
                            }
                        }
                    }
                }
                return false;
            """)
            if clicked:
                return True
        except Exception:
            pass
        xpaths = [
            "//*[contains(text(),'חיפוש מורחב')]",
            "//*[contains(@alt,'מורחב') or contains(@title,'מורחב')]",
            "//a[contains(@href,'morch') or contains(@href,'adv')]",
            "//input[contains(@value,'מורחב') or contains(@src,'morch')]",
            "//area[contains(@href,'morch') or contains(@alt,'מורחב')]",
        ]
        for xp in xpaths:
            try:
                el = driver.find_element(By.XPATH, xp)
                driver.execute_script("arguments[0].click();", el)
                return True
            except Exception:
                pass
        return False

    try:
        # ── 1. פתח אתר ──────────────────────────────────────────────────────
        driver.get(BGU_URL)
        _time.sleep(2)

        # ── 2. לחץ חיפוש מורחב (או נווט ישירות לטופס) ──────────────────────
        found_form = _click_advanced()

        if not found_form:
            frames = (driver.find_elements(By.TAG_NAME, "iframe") +
                      driver.find_elements(By.TAG_NAME, "frame"))
            for fr in frames:
                try:
                    driver.switch_to.frame(fr)
                    if _click_advanced():
                        found_form = True
                        break
                    driver.switch_to.default_content()
                except Exception:
                    driver.switch_to.default_content()

        if not found_form:
            driver.switch_to.default_content()
            for url_try in [BGU_FORM,
                            "https://bgu4u.bgu.ac.il/pls/scwp/!ann.search_adv",
                            "https://bgu4u.bgu.ac.il/pls/scwp/!app.gate?app=ann&p_type=adv"]:
                driver.get(url_try)
                _time.sleep(2)
                if (driver.find_elements(By.ID, "on_course") or
                        len(driver.find_elements(By.TAG_NAME, "select")) >= 2):
                    found_form = True
                    break

        if not found_form:
            raise RuntimeError("לא נמצא טופס חיפוש")

        _time.sleep(2)
        driver.switch_to.default_content()

        # ── 3. עבור ל-frame הטופס ───────────────────────────────────────────
        _switch_best_frame()

        # ── 4. מלא שדות ─────────────────────────────────────────────────────
        _fill("on_course_department",   department)
        _fill("on_course_degree_level", degree)
        _fill("on_course",              course_num)
        _fill("on_year",                year)

        sem_map = {"1": ["1","א","01"], "2": ["2","ב","02"], "3": ["3","קיץ","03"]}
        sem_el = None
        for loc in [(By.ID, "on_semester"), (By.NAME, "on_semester")]:
            try: sem_el = driver.find_element(*loc); break
            except Exception: pass
        if sem_el:
            _try_select(sem_el, *sem_map.get(semester, [semester]))

        # ── 5. לחץ חפש ──────────────────────────────────────────────────────
        search_clicked = False
        for attempt in ["id", "js", "xpath"]:
            try:
                if attempt == "id":
                    btn = driver.find_element(By.ID, "GOPAGE2")
                    driver.execute_script("arguments[0].click();", btn)
                elif attempt == "js":
                    driver.execute_script("goPage(2, true);")
                else:
                    btn = driver.find_element(By.XPATH,
                        "//input[@value='חפש'] | "
                        "//input[@type='button' and contains(@value,'חפש')]")
                    driver.execute_script("arguments[0].click();", btn)
                search_clicked = True
                break
            except Exception:
                pass

        if not search_clicked:
            raise RuntimeError("לא נמצא כפתור חיפוש")

        _time.sleep(3)
        _switch_main_frame()

        # ── 6. תוצאות – לחץ על הקישור הכחול ────────────────────────────────
        links = driver.find_elements(By.CSS_SELECTOR, "#courseTable tbody a")
        if not links:
            skip = {"Languages", "תפריט", "חזור", "עזרה", "logout"}
            links = [a for a in driver.find_elements(By.CSS_SELECTOR, "table a")
                     if a.text.strip() and not any(s in a.text for s in skip)]

        if not links:
            _time.sleep(5)
            return {"course_name": f"{department}.{degree}.{course_num}",
                    "schedule": [], "error": "לא נמצאו תוצאות"}

        course_name = links[0].text.strip()
        driver.execute_script("arguments[0].click();", links[0])
        _time.sleep(3)
        _switch_main_frame()

        # ── 7. סרוק טבלת שעות ───────────────────────────────────────────────
        # מבנה עמודות: 0=קבוצה, 1=סוג, 2=מרצה, 3=תא מאוחד(זמני+מקום+אופן)
        schedule = []
        for table in driver.find_elements(By.CSS_SELECTOR, "table.dataTable"):
            rows = table.find_elements(By.TAG_NAME, "tr")
            if len(rows) < 2:
                continue
            hdrs = [th.text.strip() for th in rows[0].find_elements(By.TAG_NAME, "th")]
            if "סוג" not in hdrs and "מרצה" not in hdrs:
                continue
            for row in rows[1:]:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < 4:
                    continue
                combined = cells[3].text.strip()
                if not combined:
                    continue

                t_m = _re2.search(
                    r'יום\s+[אבגדה]\s+\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}',
                    combined)
                times_raw = t_m.group(0) if t_m else ""

                loc_m = _re2.search(r'מקום לימוד:\s*(.*?)(?:\n|אופן|$)',
                                    combined, _re2.DOTALL)
                location = loc_m.group(1).strip() if loc_m else ""

                met_m = _re2.search(r'אופן לימוד:\s*(.*?)$',
                                    combined, _re2.MULTILINE)
                method = met_m.group(1).strip() if met_m else ""

                schedule.append({
                    "type":      cells[1].text.strip() if len(cells) > 1 else "",
                    "lecturer":  cells[2].text.strip() if len(cells) > 2 else "",
                    "times":     _parse_time_slots(times_raw),
                    "times_raw": times_raw,
                    "location":  location,
                    "method":    method,
                })

        _time.sleep(5)
        return {"course_name": course_name, "schedule": schedule}

    except Exception:
        _time.sleep(8)
        raise
    finally:
        driver.quit()


@app.route('/api/timetable/search', methods=['POST'])
def api_timetable_search():
    data     = request.get_json() or {}
    year     = str(data.get('year', '')).strip()
    semester = str(data.get('semester', '')).strip()

    # JS שולח שדות נפרדים: department / degree / course_num
    # או שדה אחד: course_id בפורמט "361.1.3581"
    department = str(data.get('department', '')).strip()
    degree     = str(data.get('degree', '')).strip()
    course_num = str(data.get('course_num', '')).strip()

    if not (department and degree and course_num):
        # נסה פורמט מאוחד: course_id = "361.1.3581"
        course_id = str(data.get('course_id', '')).strip()
        parts = course_id.replace(" ", "").split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            return jsonify({'error': 'פורמט שגוי. דוגמה: 361.1.3581'}), 400
        department, degree, course_num = parts

    if not all([year, semester]):
        return jsonify({'error': 'נא למלא שנה וסמסטר'}), 400

    course_id = f"{department}.{degree}.{course_num}"

    try:
        result = _scrape_timetable(department, degree, course_num, year, semester)
        result['course_id'] = course_id
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== Portal Login ====================
@app.route('/api/portal-login', methods=['POST'])
def api_portal_login():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    data     = request.get_json() or {}
    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()
    idnum    = (data.get('idnum')    or '').strip()
    if not username or not password or not idnum:
        return jsonify({'success': False, 'error': 'חסרים פרטים'}), 400
    try:
        from PORTAL import login as portal_login
        success = portal_login(username, password, idnum)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== Graduates API ====================
@app.route('/api/graduates')
def api_graduates():
    dept = request.args.get('dept', '').strip()
    year = request.args.get('year', '').strip()
    csv_path = os.path.join(BASE_PATH, 'downloads', 'Graduates', 'graduates_summary.csv')
    if not os.path.exists(csv_path):
        return jsonify({'rows': [], 'error': 'CSV not found'})
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df['שנה'] = pd.to_numeric(df['שנה'], errors='coerce')
        df['מספר_סטודנטים'] = pd.to_numeric(df['מספר_סטודנטים'], errors='coerce').fillna(0).astype(int)
        filtered = df[
            (df['מחלקה'] == dept) &
            (df['שנה'] == int(year)) &
            (df['טווח_ציונים'] != 'לא זמין')
        ].copy()
        filtered['_s'] = filtered['טווח_ציונים'].apply(
            lambda x: int(str(x).split('-')[0]) if '-' in str(x) else 0
        )
        filtered = filtered.sort_values('_s')
        rows = [{'range': r['טווח_ציונים'], 'count': int(r['מספר_סטודנטים'])}
                for _, r in filtered.iterrows()]
        return jsonify({'rows': rows})
    except Exception as e:
        return jsonify({'rows': [], 'error': str(e)})


# ---------- FORMS ----------
@app.route('/api/forms')
def api_forms():
    import re as _re_forms
    from urllib.parse import unquote as _unquote

    q = request.args.get('q', '').strip().lower()
    log_path = os.path.join(os.path.dirname(BASE_PATH), 'crawl_log.txt')

    forms = []
    seen_names = set()

    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line.startswith('PDF:'):
                    continue
                arrow = line.find(' -> ')
                if arrow == -1:
                    continue
                url = line[4:arrow].strip()
                url = _re_forms.sub(r':443', '', url)
                raw_name = url.split('/')[-1]
                if raw_name.lower().endswith('.pdf'):
                    raw_name = raw_name[:-4]
                name = _unquote(raw_name).replace('-', ' ').replace('_', ' ').strip()
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                forms.append({'name': name, 'url': url})

    if q:
        forms = [f for f in forms if q in f['name'].lower()]

    return jsonify({'forms': forms[:60], 'total': len(forms)})


# ==================== Main ====================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  BGU Manager - Student Portal")
    print("="*50)
    print("  Open: http://localhost:5001")
    print("="*50 + "\n")
    app.run(debug=False, port=5001, host='0.0.0.0', threaded=True)
