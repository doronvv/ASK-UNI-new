"""
BGUManager.py
=============
בוט BGU – ממשק ווב (Streamlit) + Google Gemini + ChromaDB.
"""

import re as _re_links
import sys
from pathlib import Path
import streamlit.components.v1 as _st_components

import chromadb
import google.generativeai as genai
import pandas as pd
import streamlit as st

# --- שליפת מפתח API מאובטחת ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    # מפתח גיבוי להרצה לוקאלית על המחשב שלך (הכנס את המפתח שלך כאן במקום 'חסוי')
    GOOGLE_API_KEY = "חסוי"

MODEL_NAME = "gemini-1.5-flash"  # תוקן למודל קיים ויציב

# --- תיקון נתיבים (הוצאה מתיקיית ה-.venv) ---
DATA_DIR      = Path(__file__).parent / "data"
CHROMA_DIR    = Path(__file__).parent / "chroma_db_storage"
COLLECTION    = "bgu_knowledge"
GRADUATES_CSV = DATA_DIR / "graduates_summary.csv" # תוקן
CRAWL_LOG     = Path(__file__).parent / "crawl_log.txt"

_csv_cache: dict[str, pd.DataFrame] = {}


# ── נתונים ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_graduates() -> pd.DataFrame | None:
    if not GRADUATES_CSV.exists():
        return None
    for enc in ("utf-8-sig", "utf-8", "cp1255", "latin-1"):
        try:
            df = pd.read_csv(GRADUATES_CSV, encoding=enc)
            df["שנה"] = pd.to_numeric(df["שנה"], errors="coerce").astype("Int64")
            df["מספר_סטודנטים"] = pd.to_numeric(df["מספר_סטודנטים"], errors="coerce").fillna(0).astype(int)
            return df
        except Exception:
            continue
    return None


def _load_csv(name: str) -> pd.DataFrame | None:
    if name in _csv_cache:
        return _csv_cache[name]
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        return None
    for enc in ("utf-8-sig", "cp1255", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            _csv_cache[name] = df
            return df
        except Exception:
            continue
    return None


# ── כלים ─────────────────────────────────────────────────────────────────────

_S = genai.protos.Schema
_T = genai.protos.Type

TOOLS = [
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="search_forms",
                description=(
                    "חיפוש טפסים של האוניברסיטה לפי שם או נושא. "
                    "השתמש כשהמשתמש שואל על טופס, מסמך, בקשה, הצהרה או אישור."
                ),
                parameters=_S(
                    type=_T.OBJECT,
                    properties={
                        "query": _S(type=_T.STRING, description="שם הטופס או נושא החיפוש"),
                    },
                    required=["query"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="query_dataframe",
                description=(
                    "שאילתת pandas – ממוצע, סינון, מיון, ספירה. "
                    "עמודות ציון כניסה: סכם_הנדסה/סכם_כמותי/סכם. "
                    "עמודות קורסים: 2025A/2025B/2024A/2024B."
                ),
                parameters=_S(
                    type=_T.OBJECT,
                    properties={
                        "file_name": _S(
                            type=_T.STRING,
                            description=(
                                "שם קובץ ללא .csv: "
                                "bgu_admission, bgu_admission_requirements, bgu_admission_complete, "
                                "g_all_courses_with_grades, grades, grades_machinery, machinery2_with_grades, "
                                "bgu_scholarships_new_3, Projects_Classified, PROJECT_WEB1, "
                                "army_results, courses_fixed, machinery, g_all_courses"
                            ),
                        ),
                        "operation": _S(
                            type=_T.STRING,
                            description="ביטוי pandas עם df. לדוגמה: df[df['שם_המסלול'].str.contains('חשמל')]",
                        ),
                    },
                    required=["file_name", "operation"],
                ),
            ),
        ]
    )
]

SYSTEM_PROMPT = (
    "אתה AskUni – עוזר מידע של אוניברסיטת בן גוריון. ענה תמיד בעברית.\n"
    "כלל יסוד: ענה אך ורק על בסיס קטעי המידע שסופקו. אל תמציא עובדות.\n"
    "קישורים: כל קישור שמופיע בקטע – כלול אותו בתשובה כ: 🔗 [תיאור](URL).\n"
    "אי-ודאות: אם הקטעים לא מכסים את השאלה – כתוב 'לא נמצא מידע מספיק. פנה ל: https://www.bgu.ac.il' .\n"
    "query_dataframe: לחישובים (ממוצע/סינון/מיון). עמודות ציון כניסה: סכם_הנדסה/סכם_כמותי/סכם.\n"
    "search_forms: לשאלות על טופס/מסמך/בקשה/אישור."
)


# ── לוגיקה ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_chroma() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION)


_MAX_DOC_CHARS    = 500   
_RAG_N            = 5     
_MAX_PROMPT_CHARS = 3500  


def _retrieve_context(col: chromadb.Collection, query: str) -> str:
    try:
        results = col.query(query_texts=[query], n_results=_RAG_N)
    except Exception as e:
        return f"**הקשר מ-ChromaDB:** שגיאת אחזור – {e}"

    docs  = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    if not docs:
        return (
            "**הקשר מ-ChromaDB:** לא נמצאו תוצאות.\n"
            "לפרטים פנה לאתר הרשמי: https://www.bgu.ac.il"
        )

    parts = ["**קטעי מידע רלוונטיים מהמסד (BGU):**"]
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        text = str(doc or "")[:_MAX_DOC_CHARS]
        if len(str(doc or "")) > _MAX_DOC_CHARS:
            text += "…"
        url = (meta or {}).get("url", "") or ""
        url_line = f"\nקישור: {url}" if url else ""
        parts.append(
            f"\n[קטע {i} | סוג: {(meta or {}).get('type', '')}]\n"
            f"{text}{url_line}"
        )
    return "\n".join(parts)


def run_search(col: chromadb.Collection, query: str, n: int, type_filter: str) -> str:
    where = {"type": type_filter} if type_filter else None
    try:
        results = col.query(query_texts=[query], n_results=n, where=where)
    except Exception as e:
        return f"שגיאת חיפוש: {e}"
    docs  = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    if not docs:
        return "לא נמצאו תוצאות."
    parts = []
    for doc, meta, dist in zip(docs, metas, dists):
        score = round(1 - dist, 3) if dist is not None else "?"
        snippet = doc[:_MAX_DOC_CHARS] + ("…" if len(doc) > _MAX_DOC_CHARS else "")
        parts.append(f"[רלוונטיות: {score}] [{meta.get('type', '')}]\n{snippet}")
    return "\n\n---\n".join(parts)


def run_forms_search(query: str) -> str:
    if not CRAWL_LOG.exists():
        return "קובץ יומן הטפסים לא נמצא."
    from urllib.parse import unquote as _unquote
    keywords = [w.strip() for w in query.split() if len(w.strip()) > 1]
    matches: list[tuple[str, str]] = []
    seen: set[str] = set()
    with open(CRAWL_LOG, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("PDF:"):
                continue
            arrow = line.find(" -> ")
            if arrow == -1:
                continue
            url = line[4:arrow].strip()
            url = _re_links.sub(r':443', '', url)
            raw = url.split('/')[-1]
            if raw.lower().endswith('.pdf'):
                raw = raw[:-4]
            name = _unquote(raw).replace('-', ' ').replace('_', ' ').strip()
            if not name or name in seen:
                continue
            seen.add(name)
            name_lower = name.lower()
            if any(kw in name_lower for kw in keywords):
                matches.append((name, url))
    if not matches:
        return f"לא נמצאו טפסים התואמים ל: {query}"
    lines = [f"נמצאו {len(matches)} טפסים רלוונטיים:"]
    for name, url in matches[:8]:
        lines.append(f"• {name}: {url}")
    return "\n".join(lines)


def run_query(file_name: str, operation: str) -> str:
    df = _load_csv(file_name)
    if df is None:
        available = sorted(p.stem for p in DATA_DIR.glob("*.csv"))
        return f"קובץ '{file_name}' לא קיים. זמינים: {available}"
    df = df.copy()
    try:
        result = eval(operation, {"df": df, "pd": pd})  # noqa: S307
        if isinstance(result, pd.DataFrame):
            cap = 60
            prefix = f"[מציג {cap} מתוך {len(result)} שורות]\n" if len(result) > cap else ""
            return prefix + result.head(cap).to_string(index=False)
        if isinstance(result, pd.Series):
            return result.to_string()
        return str(result)
    except Exception as exc:
        return f"שגיאה: {exc}"


def _dispatch(fn_name: str, fn_args: dict) -> str:
    if fn_name == "search_forms":
        return run_forms_search(fn_args.get("query", ""))
    if fn_name == "query_dataframe":
        fname = fn_args.get("file_name", "")
        return run_query(fname, fn_args.get("operation", ""))
    return f"כלי לא מוכר: {fn_name}"


def _compress(text: str) -> str:
    import re as _re
    text = _re.sub(r'[ \t]+', ' ', text)
    text = _re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


_RETRY_WAITS = [5, 15, 30]  


def _is_rate_limit(err: Exception) -> bool:
    s = str(err)
    return "429" in s or "quota" in s.lower() or "rate" in s.lower()


def ask_gemini(col: chromadb.Collection, user_text: str,
               model: genai.GenerativeModel) -> str:
    import time as _time

    rag_context = _retrieve_context(col, user_text)

    prompt = _compress(f"{rag_context}\n\n---\nשאלת המשתמש: {user_text}")
    if len(prompt) > _MAX_PROMPT_CHARS:
        prompt = (prompt[:_MAX_PROMPT_CHARS]
                  + "\n\n[חתוך. ענה על בסיס המידע הקיים.]")

    last_err: Exception | None = None
    for attempt, wait in enumerate([0] + _RETRY_WAITS):
        if wait:
            _time.sleep(wait)
        try:
            chat = model.start_chat()
            response = chat.send_message(prompt)

            while True:
                fn_calls = [p.function_call for p in response.parts
                            if p.function_call.name]
                if not fn_calls:
                    break
                tool_parts = []
                for fc in fn_calls:
                    result = _dispatch(fc.name, dict(fc.args))
                    tool_parts.append(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=fc.name,
                                response={"result": result},
                            )
                        )
                    )
                response = chat.send_message(tool_parts)

            try:
                return response.text
            except Exception:
                return "".join(p.text for p in response.parts if p.text)

        except Exception as e:
            last_err = e
            if not _is_rate_limit(e):
                raise  
            
    raise last_err  # type: ignore


# ── מערכת שעות – Selenium ─────────────────────────────────────────────

import re as _re_tt

_DAY_NAMES_TT = {'א': 'ראשון', 'ב': 'שני', 'ג': 'שלישי', 'ד': 'רביעי', 'ה': 'חמישי'}
_BGU_START    = "https://bgu4u.bgu.ac.il/pls/scwp/!app.gate?app=ann"
_BGU_FORM     = "https://bgu4u.bgu.ac.il/pls/scwp/!app.ann?lang=he"


def _parse_time_slots_tt(txt: str) -> list:
    slots = []
    if not txt:
        return slots
    for m in _re_tt.finditer(
            r'([אבגדה](?:[,\s]*[אבגדה])*)\s+(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})', txt):
        days  = _re_tt.findall(r'[אבגדה]', m.group(1))
        start = m.group(2).zfill(5)
        end   = m.group(3).zfill(5)
        for d in days:
            slots.append({'day': d, 'day_name': _DAY_NAMES_TT.get(d, d),
                          'start': start, 'end': end})
    return slots


def _scrape_timetable_visible(department: str, degree: str, course_num: str,
                               year: str, semester: str) -> dict:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select
    from selenium.webdriver.chrome.options import Options
    import time as _t
    import re as _r

    opts = Options()
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--disable-infobars")
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])
    
    # --- הוספת Headless כדי שיעבוד בענן (Streamlit Cloud) ---
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    
    driver = webdriver.Chrome(options=opts)

    def _try_sel(el, *vals):
        sel = Select(el)
        for v in vals:
            try: sel.select_by_value(v); return True
            except Exception: pass
            for opt in sel.options:
                if v in (opt.get_attribute("value") or "") or v in opt.text:
                    try: sel.select_by_value(opt.get_attribute("value")); return True
                    except Exception:
                        try: opt.click(); return True
                        except Exception: pass
        return False

    def _fill(fid, val):
        for loc in [(By.ID, fid), (By.NAME, fid)]:
            try:
                el = driver.find_element(*loc)
                if el.tag_name.lower() == "select":
                    return _try_sel(el, val)
                el.clear(); el.send_keys(val); return True
            except Exception: pass
        return False

    def _best_frame():
        driver.switch_to.default_content()
        best_n, best_i = len(driver.find_elements(By.TAG_NAME, "select")), None
        frs = (driver.find_elements(By.TAG_NAME, "iframe") +
               driver.find_elements(By.TAG_NAME, "frame"))
        for i, fr in enumerate(frs):
            try:
                driver.switch_to.frame(fr)
                n = len(driver.find_elements(By.TAG_NAME, "select"))
                driver.switch_to.default_content()
                if n > best_n: best_n, best_i = n, i
            except Exception: driver.switch_to.default_content()
        if best_i is not None:
            frs = (driver.find_elements(By.TAG_NAME, "iframe") +
                   driver.find_elements(By.TAG_NAME, "frame"))
            driver.switch_to.frame(frs[best_i])

    def _main_frame():
        driver.switch_to.default_content()
        try: driver.switch_to.frame("main")
        except Exception: pass

    def _click_adv():
        try:
            ok = driver.execute_script("""
                var kw=['מורחב','morch','adv','Advanced'];
                var tags=['a','input','button','area','img','span','div','td'];
                for(var t=0;t<tags.length;t++){
                    var els=document.getElementsByTagName(tags[t]);
                    for(var i=0;i<els.length;i++){
                        var e=els[i];
                        var h=(e.textContent||'')+(e.value||'')+(e.alt||'')+(e.title||'')+(e.href||'')+(e.src||'');
                        for(var k=0;k<kw.length;k++)if(h.indexOf(kw[k])!==-1){e.click();return true;}
                    }
                }return false;
            """)
            if ok: return True
        except Exception: pass
        for xp in ["//*[contains(text(),'חיפוש מורחב')]",
                   "//a[contains(@href,'morch') or contains(@href,'adv')]",
                   "//area[contains(@href,'morch') or contains(@alt,'מורחב')]"]:
            try:
                driver.execute_script("arguments[0].click();",
                                      driver.find_element(By.XPATH, xp))
                return True
            except Exception: pass
        return False

    try:
        driver.get(_BGU_START)
        _t.sleep(2)

        found = _click_adv()
        if not found:
            frs = (driver.find_elements(By.TAG_NAME, "iframe") +
                   driver.find_elements(By.TAG_NAME, "frame"))
            for fr in frs:
                try:
                    driver.switch_to.frame(fr)
                    if _click_adv(): found = True; break
                    driver.switch_to.default_content()
                except Exception: driver.switch_to.default_content()

        if not found:
            driver.switch_to.default_content()
            for url in [_BGU_FORM,
                        "https://bgu4u.bgu.ac.il/pls/scwp/!ann.search_adv"]:
                driver.get(url); _t.sleep(2)
                if (driver.find_elements(By.ID, "on_course") or
                        len(driver.find_elements(By.TAG_NAME, "select")) >= 2):
                    found = True; break

        if not found:
            raise RuntimeError("לא נמצא טופס חיפוש")

        _t.sleep(2)
        driver.switch_to.default_content()

        _best_frame()

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
            _try_sel(sem_el, *sem_map.get(semester, [semester]))

        clicked = False
        for att in ["id", "js", "xpath"]:
            try:
                if att == "id":
                    btn = driver.find_element(By.ID, "GOPAGE2")
                    driver.execute_script("arguments[0].click();", btn)
                elif att == "js":
                    driver.execute_script("goPage(2, true);")
                else:
                    btn = driver.find_element(By.XPATH,
                        "//input[@value='חפש'] | "
                        "//input[@type='button' and contains(@value,'חפש')]")
                    driver.execute_script("arguments[0].click();", btn)
                clicked = True; break
            except Exception: pass

        if not clicked:
            raise RuntimeError("לא נמצא כפתור חיפוש")

        _t.sleep(3)
        _main_frame()

        links = driver.find_elements(By.CSS_SELECTOR, "#courseTable tbody a")
        if not links:
            skip = {"Languages", "תפריט", "חזור", "עזרה", "logout"}
            links = [a for a in driver.find_elements(By.CSS_SELECTOR, "table a")
                     if a.text.strip() and not any(s in a.text for s in skip)]

        if not links:
            _t.sleep(5); driver.quit()
            return {"course_name": f"{department}.{degree}.{course_num}",
                    "schedule": [], "error": "לא נמצאו תוצאות"}

        course_name = links[0].text.strip()
        driver.execute_script("arguments[0].click();", links[0])
        _t.sleep(3)
        driver.switch_to.default_content()
        try: driver.switch_to.frame("main")
        except Exception: pass

        schedule = []
        for table in driver.find_elements(By.CSS_SELECTOR, "table.dataTable"):
            rows = table.find_elements(By.TAG_NAME, "tr")
            if len(rows) < 2: continue
            hdrs = [th.text.strip() for th in rows[0].find_elements(By.TAG_NAME, "th")]
            if "סוג" not in hdrs and "מרצה" not in hdrs: continue
            for row in rows[1:]:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < 4: continue
                combined = cells[3].text.strip()
                if not combined: continue
                t_m = _r.search(
                    r'יום\s+[אבגדה]\s+\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}', combined)
                times_raw = t_m.group(0) if t_m else ""
                loc_m = _r.search(r'מקום לימוד:\s*(.*?)(?:\n|אופן|$)', combined, _r.DOTALL)
                location = loc_m.group(1).strip() if loc_m else ""
                met_m = _r.search(r'אופן לימוד:\s*(.*?)$', combined, _r.MULTILINE)
                method = met_m.group(1).strip() if met_m else ""
                schedule.append({
                    "type":      cells[1].text.strip() if len(cells) > 1 else "",
                    "lecturer":  cells[2].text.strip() if len(cells) > 2 else "",
                    "times":     _parse_time_slots_tt(times_raw),
                    "times_raw": times_raw,
                    "location":  location,
                    "method":    method,
                })

        _t.sleep(5)
        return {"course_name": course_name, "schedule": schedule}

    except Exception:
        _t.sleep(8)
        raise
    finally:
        driver.quit()


# ── ממשק Streamlit ───────────────────────────────────────────────────────────

FACULTY_IMAGE = Path(__file__).parent / "אנשי סגל.png" # תוקן להשתמש בתיקייה הראשית

TOPICS = [
    ("🎓", "קבלה",     "ציוני כניסה ודרישות קבלה"),
    ("📊", "ציונים",   "ממוצעי קורסים וציוני סיום"),
    ("💰", "מלגות",    "מלגות זמינות ותנאי קבלה"),
    ("🔬", "פרויקטים", "פרויקטי גמר ומחקר"),
    ("📚", "קורסים",   "מידע על קורסים"),
    ("🪖", "מילואים",  "זכויות משרתי מילואים"),
]


def _grades_nav_ui() -> None:
    gdf = load_graduates()
    if gdf is None or gdf.empty:
        st.warning("אין נתונים. נא לוודא שקובץ graduates_summary.csv נמצא בתיקיית data.")
        return

    view = st.session_state.get("grade_view", "depts")

    if view == "depts":
        st.markdown("### בחר תואר")
        depts = sorted(gdf["מחלקה"].dropna().unique().tolist())
        for row_start in range(0, len(depts), 3):
            row_depts = depts[row_start:row_start + 3]
            cols = st.columns(3)
            for j, dept in enumerate(row_depts):
                with cols[j]:
                    if st.button(dept, key=f"gd_{dept}", use_container_width=True):
                        st.session_state["grade_dept_nav"] = dept
                        st.session_state["grade_view"] = "years"
                        st.rerun()

    elif view == "years":
        dept = st.session_state.get("grade_dept_nav", "")
        if st.button("← חזור לרשימת תארים", key="grade_back_depts"):
            st.session_state["grade_view"] = "depts"
            st.rerun()
        st.markdown(f"### {dept}")
        st.markdown("בחר שנה:")
        yr_cols = st.columns(4)
        for i, yr in enumerate([2024, 2023, 2022, 2021]):
            with yr_cols[i]:
                if st.button(str(yr), key=f"gyr_{yr}",
                             use_container_width=True, type="primary"):
                    st.session_state["grade_year_nav"] = yr
                    st.session_state["grade_view"] = "table"
                    st.rerun()

    elif view == "table":
        dept = st.session_state.get("grade_dept_nav", "")
        year = st.session_state.get("grade_year_nav")

        if st.button("← חזור לשנים", key="grade_back_years"):
            st.session_state["grade_view"] = "years"
            st.rerun()

        st.markdown(f"### {dept} – {year}")

        filtered = gdf[
            (gdf["מחלקה"] == dept) &
            (gdf["שנה"]   == year) &
            (gdf["טווח_ציונים"] != "לא זמין")
        ].copy()

        if filtered.empty:
            st.info("אין נתונים לתואר ושנה אלו.")
        else:
            filtered["_s"] = filtered["טווח_ציונים"].apply(
                lambda x: int(str(x).split("-")[0]) if "-" in str(x) else 0
            )
            filtered = filtered.sort_values(by="_s")
            chart_df = filtered[["טווח_ציונים", "מספר_סטודנטים"]].set_index("טווח_ציונים")
            st.bar_chart(chart_df, y="מספר_סטודנטים", height=400)
            st.dataframe(
                filtered[["טווח_ציונים", "מספר_סטודנטים"]].reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )

        st.markdown("**החלף שנה:**")
        yr_cols = st.columns(4)
        for i, yr in enumerate([2024, 2023, 2022, 2021]):
            with yr_cols[i]:
                btn_type = "primary" if yr == year else "secondary"
                if st.button(str(yr), key=f"gyrb_{yr}",
                             use_container_width=True, type=btn_type):
                    st.session_state["grade_year_nav"] = yr
                    st.rerun()


def _render_answer(answer: str) -> None:
    def _link_html(label: str, url: str) -> str:
        return (
            f'<span style="display:inline-flex;align-items:center;gap:5px;flex-wrap:wrap">'
            f'<a href="{url}" target="_blank" '
            f'style="color:#e8973a;font-weight:700;text-decoration:none">{label}</a>'
            f'<button data-copy="{url}" '
            f'style="background:#f5f6fa;border:1px solid #e0e0e0;border-radius:6px;'
            f'padding:2px 8px;font-size:11px;cursor:pointer;font-family:inherit;'
            f'transition:background .15s" '
            f'onclick="var u=this.getAttribute(\'data-copy\');'
            f'navigator.clipboard.writeText(u);'
            f'var b=this;b.textContent=\'✓ הועתק\';'
            f'setTimeout(function(){{b.textContent=\'📋 העתק\';}},1500)">'
            f'📋 העתק</button>'
            f'</span>'
        )

    links: list[str] = []

    def _store(label: str, url: str) -> str:
        links.append(_link_html(label, url))
        return f'\x00LINK{len(links)-1}\x00'

    html = answer
    html = _re_links.sub(
        r'\[([^\]]+)\]\((https?://[^)\s]+)\)',
        lambda m: _store(m.group(1), m.group(2)),
        html,
    )
    html = _re_links.sub(
        r'(?<![=("\'`>])(https?://[^\s<>"\')\]]+)',
        lambda m: _store(m.group(0), m.group(0)),
        html,
    )
    html = _re_links.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = html.replace('\n', '<br>')
    html = _re_links.sub(r'\x00LINK(\d+)\x00', lambda m: links[int(m.group(1))], html)

    lines = answer.count('\n')
    height = max(80, lines * 22 + 60)
    _st_components.html(
        f'<div style="font-family:\'Segoe UI\',Arial,sans-serif;font-size:14px;'
        f'line-height:1.75;direction:rtl;text-align:right;color:#333;padding:4px 0">'
        f'{html}</div>',
        height=height,
        scrolling=False,
    )


def run_ui() -> None:
    st.set_page_config(page_title="עוזר BGU", layout="wide", page_icon="🎓")

    st.markdown("""
    <style>
        .stApp { direction: rtl; text-align: right; }
        .stChatMessage { direction: rtl; text-align: right; }
        .stChatInput textarea { direction: rtl; text-align: right; }
        p, h1, h2, h3, li { text-align: right; direction: rtl; }
        .topic-btn { width: 100%; margin-bottom: 4px; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 👥 אנשי סגל")
        if FACULTY_IMAGE.exists():
            st.image(str(FACULTY_IMAGE), use_container_width=True)
        st.caption("חוקרים, מרצים ועובדים באוניברסיטת בן גוריון")
        if st.button("שאל על אנשי סגל", key="btn_people", use_container_width=True):
            st.session_state.pending = "ספר לי על אנשי הסגל באוניברסיטת בן גוריון"

        st.divider()
        st.markdown("### נושאים נוספים")
        TOPIC_LINKS = {
            "מלגות":  [("לינק לאתר הרשמי של המלגות", "https://www.bgu.ac.il/welcome/ba/scholarship-lobby/")],
            "מילואים": [("אתר המילואים הרשמי", "https://www.bgu.ac.il/u/academic-affairs/dekanat/miluim/")],
            "קבלה":   [("תנאי קבלה", "https://www.bgu.ac.il/welcome/ba/reception-section-lobby/?semesters=012027")],
        }
        for icon, label, desc in TOPICS:
            with st.expander(f"{icon} {label}"):
                st.caption(desc)
                for link_text, link_url in TOPIC_LINKS.get(label, []):
                    st.markdown(f"[{link_text}]({link_url})")
                if st.button(f"שאל על {label}", key=f"btn_{label}", use_container_width=True):
                    st.session_state.pending = f"ספר לי על {label} באוניברסיטת בן גוריון"

    st.title("🎓 עוזר אוניברסיטת בן גוריון")

    with st.expander("📊 ציוני סוף תארים", expanded=False):
        _grades_nav_ui()

    tab_chat, tab_tt, tab_portal = st.tabs(
        ["💬 שיחה", "🕐 מערכת שעות", "🔐 פורטל"]
    )

    with tab_chat:
        try:
            col = load_chroma()
        except Exception as e:
            st.error(f"ChromaDB לא נמצא – ודא שהתיקייה chroma_db_storage קיימת במקביל לקובץ זה.\n\n{e}")
            st.stop()

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "pending" not in st.session_state:
            st.session_state.pending = ""

        if "gemini_model" not in st.session_state:
            genai.configure(api_key=GOOGLE_API_KEY)
            st.session_state.gemini_model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                tools=TOOLS,
                system_instruction=SYSTEM_PROMPT,
            )

        if st.button("🗑️ נקה שיחה", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    _render_answer(msg["content"])
                else:
                    st.markdown(msg["content"])

        _st_components.html("""
<script>
(function() {
    const FULL = "שאל שאלה על BGU...";
    let idx = 0, growing = true;

    function getInput() {
        return document.querySelector('[data-testid="stChatInput"] textarea')
            || document.querySelector('textarea[placeholder]');
    }

    function tick() {
        const el = getInput();
        if (!el) { setTimeout(tick, 150); return; }

        if (growing) {
            idx++;
            if (idx >= FULL.length) { growing = false; setTimeout(tick, 1800); return; }
        } else {
            idx--;
            if (idx <= 0) { growing = true; setTimeout(tick, 500); return; }
        }
        el.setAttribute("placeholder", FULL.substring(0, idx));
        setTimeout(tick, growing ? 90 : 45);
    }
    setTimeout(tick, 400);
})();
</script>
""", height=0)
        pending = st.session_state.pop("pending", "")
        prompt = st.chat_input("שאל שאלה על BGU...") or pending
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("מחפש..."):
                    try:
                        answer = ask_gemini(col, prompt, st.session_state.gemini_model)
                    except Exception as e:
                        err = str(e)
                        if "quota" in err.lower() or "429" in err:
                            answer = "⚠️ הגעת למגבלת השימוש ב-Gemini. המתן כמה שניות ונסה שוב."
                        else:
                            answer = f"⚠️ שגיאה: {err}"
                _render_answer(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

    with tab_tt:
        st.subheader("חיפוש מערכת שעות קורס")
        st.caption("ימלא את הטופס, ילחץ חפש ויביא את שעות הקורס.")

        _st_components.html("""
<!DOCTYPE html>
<html dir="rtl">
<head>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: Arial, sans-serif; background: transparent; direction: rtl; }

  .demo-wrap {
    background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
    border: 1px solid #c5d3f0;
    border-radius: 12px;
    padding: 14px 18px 16px;
    margin-bottom: 6px;
  }
  .demo-title {
    font-size: 13px;
    color: #3c4f7c;
    font-weight: 700;
    margin-bottom: 12px;
    letter-spacing: 0.3px;
  }
  .fields-row {
    display: flex;
    gap: 18px;
    align-items: flex-end;
    flex-wrap: wrap;
  }
  .field-box {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
  }
  .field-label {
    font-size: 11px;
    color: #5f6d8a;
    margin-bottom: 5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .fake-input {
    min-width: 72px;
    padding: 7px 12px;
    border: 2px solid #b0c0e8;
    border-radius: 8px;
    background: #ffffff;
    font-size: 20px;
    font-weight: 700;
    color: #1a2a5e;
    min-height: 42px;
    line-height: 1.3;
    transition: border-color 0.3s;
    display: flex;
    align-items: center;
  }
  .fake-input.active {
    border-color: #4169e1;
    box-shadow: 0 0 0 3px rgba(65,105,225,0.15);
  }
  .cursor {
    display: inline-block;
    width: 2px;
    height: 22px;
    background: #4169e1;
    margin-right: 1px;
    vertical-align: middle;
    border-radius: 1px;
    animation: blink 0.6s step-end infinite;
  }
  .cursor.hidden { display: none; }
  @keyframes blink { 50% { opacity: 0; } }

  .dot-sep {
    font-size: 22px;
    font-weight: 700;
    color: #4169e1;
    padding-bottom: 8px;
    opacity: 0;
    transition: opacity 0.3s;
  }
  .dot-sep.show { opacity: 1; }

  .status-line {
    font-size: 11px;
    color: #7a8ab0;
    margin-top: 10px;
    min-height: 16px;
  }
</style>
</head>
<body>
<div class="demo-wrap">
  <div class="demo-title">🖊 דוגמה – כך ממלאים מספר קורס (מחלקה.תואר.קורס):</div>
  <div class="fields-row">

    <div class="field-box">
      <span class="field-label">מחלקה</span>
      <div class="fake-input" id="box-dept">
        <span id="txt-dept"></span><span class="cursor hidden" id="cur-dept"></span>
      </div>
    </div>

    <div class="dot-sep" id="dot1">.</div>

    <div class="field-box">
      <span class="field-label">תואר</span>
      <div class="fake-input" id="box-deg">
        <span id="txt-deg"></span><span class="cursor hidden" id="cur-deg"></span>
      </div>
    </div>

    <div class="dot-sep" id="dot2">.</div>

    <div class="field-box">
      <span class="field-label">מספר קורס</span>
      <div class="fake-input" id="box-cnum" style="min-width:100px;">
        <span id="txt-cnum"></span><span class="cursor hidden" id="cur-cnum"></span>
      </div>
    </div>

  </div>
  <div class="status-line" id="status-line"></div>
</div>

<script>
var DEPT  = "361";
var DEG   = "1";
var CNUM  = "3581";
var CHAR_DELAY   = 100;
var FIELD_GAP    = 500;
var PAUSE_FULL   = 10000;
var CLEAR_PAUSE  = 400;

var fields = [
  { txt: "txt-dept",  cur: "cur-dept",  box: "box-dept",  dot: null,   value: DEPT },
  { txt: "txt-deg",   cur: "cur-deg",   box: "box-deg",   dot: "dot1", value: DEG  },
  { txt: "txt-cnum",  cur: "cur-cnum",  box: "box-cnum",  dot: "dot2", value: CNUM }
];

function $(id){ return document.getElementById(id); }

function clearAll() {
  fields.forEach(function(f){
    $(f.txt).textContent = "";
    $(f.cur).classList.add("hidden");
    $(f.box).classList.remove("active");
    if(f.dot) $(f.dot).classList.remove("show");
  });
  $("status-line").textContent = "";
}

function typeField(fi, ci, onDone) {
  var f = fields[fi];
  var box = $(f.box);
  var txt = $(f.txt);
  var cur = $(f.cur);

  if(ci === 0) {
    if(f.dot) $(f.dot).classList.add("show");
    box.classList.add("active");
    cur.classList.remove("hidden");
    txt.textContent = "";
  }

  if(ci >= f.value.length) {
    cur.classList.add("hidden");
    box.classList.remove("active");
    setTimeout(function(){ onDone(); }, FIELD_GAP);
    return;
  }

  txt.textContent += f.value[ci];
  setTimeout(function(){ typeField(fi, ci+1, onDone); }, CHAR_DELAY);
}

function typeAll(fi, cb) {
  if(fi >= fields.length){ cb(); return; }
  typeField(fi, 0, function(){ typeAll(fi+1, cb); });
}

function startCountdown(sec) {
  if(sec <= 0){ $("status-line").textContent = ""; return; }
  $("status-line").textContent = "סבב חדש עוד " + sec + " שניות...";
  setTimeout(function(){ startCountdown(sec-1); }, 1000);
}

function loop() {
  clearAll();
  setTimeout(function(){
    typeAll(0, function(){
      $("status-line").textContent = "✓  361.1.3581";
      startCountdown(10);
      setTimeout(function(){
        loop();
      }, PAUSE_FULL + CLEAR_PAUSE);
    });
  }, 300);
}

loop();
</script>
</body>
</html>
""", height=0)

        _qp   = st.query_params
        _dept = _qp.get("dept",  "")
        _deg  = _qp.get("deg",   "")
        _cnum = _qp.get("cnum",  "")
        _year = _qp.get("yr",    "2026")
        _sem  = _qp.get("sem",   "1")
        _sid  = _qp.get("sid",   "")

        _sem_disp  = {"1": "א'", "2": "ב'", "3": "קיץ"}.get(_sem, "א'")
        _pre       = "true" if _sid else "false"

        _st_components.html(f"""<!DOCTYPE html>
<html lang="he" dir="rtl"><head><meta charset="UTF-8"><style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:system-ui,sans-serif;background:transparent;padding:8px 2px 6px;}}
.card{{background:#fff;border-radius:14px;padding:1.3rem 1.6rem;
       box-shadow:0 2px 14px rgba(0,0,0,0.08);}}
h2{{font-size:19px;font-weight:600;margin-bottom:3px;}}
.sub{{font-size:13px;color:#888;margin-bottom:1.3rem;}}
.row{{display:flex;align-items:flex-end;gap:10px;flex-wrap:wrap;}}
.add-btn{{display:flex;align-items:center;gap:8px;background:#f5a623;color:#fff;
          border:none;border-radius:10px;padding:10px 18px;font-size:15px;
          font-weight:600;cursor:pointer;white-space:nowrap;transition:background .15s;}}
.add-btn:hover{{background:#e0941a;}}
.fgrp{{display:flex;align-items:flex-end;gap:6px;flex-direction:row-reverse;}}
.fw{{display:flex;flex-direction:column;align-items:center;gap:4px;}}
.fl{{font-size:12px;color:#999;white-space:nowrap;}}
.ti{{text-align:center;border:1.5px solid #ddd;border-radius:8px;
     padding:7px 8px;font-size:15px;font-weight:500;color:#222;
     background:#fff;outline:none;transition:border-color .2s,box-shadow .2s;
     caret-color:#f5a623;}}
.ti.active{{border-color:#4f8ef7;box-shadow:0 0 0 3px rgba(79,142,247,.15);}}
.ti:focus{{border-color:#4f8ef7;box-shadow:0 0 0 3px rgba(79,142,247,.15);}}
.sep{{font-size:14px;color:#bbb;padding-bottom:9px;}}
.dash{{font-size:19px;color:#bbb;padding-bottom:8px;}}
.ctrl{{margin-top:1rem;display:flex;align-items:center;gap:14px;}}
#tog{{font-size:13px;color:#555;background:none;border:1px solid #ddd;
      border-radius:8px;padding:6px 14px;cursor:pointer;}}
#tog:hover{{background:#f5f5f5;}}
.stat{{font-size:13px;color:#888;display:flex;align-items:center;gap:6px;}}
.dot{{width:8px;height:8px;border-radius:50%;transition:background .3s;}}
.dot.typing{{background:#4f8ef7;animation:pulse .8s ease-in-out infinite;}}
.dot.waiting{{background:#f5a623;}}
.dot.stopped{{background:#ccc;}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
.err{{font-size:12px;color:#e53e3e;margin-top:8px;min-height:16px;}}
.cyc-bar{{margin-top:14px;padding-top:10px;border-top:2px solid #f0f0f0;}}
.cyc-track{{position:relative;height:10px;background:#ebebeb;border-radius:5px;margin:0 4px;}}
.cyc-fill{{height:100%;width:0%;background:linear-gradient(90deg,#4f8ef7,#f5a623);border-radius:5px;}}
.cyc-thumb{{position:absolute;top:-5px;left:0%;width:20px;height:20px;background:#fff;border:2.5px solid #4f8ef7;border-radius:50%;transform:translateX(-50%);box-shadow:0 1px 4px rgba(0,0,0,.15);}}
.cyc-start{{position:absolute;left:0;top:-8px;width:4px;height:26px;background:#e53935;border-radius:2px;z-index:3;box-shadow:0 0 6px rgba(229,57,53,.5);}}
.cyc-lbl{{display:flex;justify-content:space-between;font-size:10px;color:#bbb;margin-top:5px;padding:0 4px;}}
.cyc-start-tip{{position:absolute;left:6px;top:-20px;font-size:9px;color:#e53935;font-weight:700;white-space:nowrap;}}
</style></head><body>
<div class="card">
  <h2>🗓️ מערכת שעות</h2>
  <p class="sub">בנה את מערכת השעות שלך – הוסף קורסים לפי מחלקה, תואר ומספר</p>
  <div class="row">
    <button class="add-btn" id="btn-add">
      <span style="font-size:20px;font-weight:400">+</span> הוסף קורס
    </button>
    <div class="fgrp">
      <div class="fw"><span class="fl">שנה</span>
        <input class="ti" id="f-year"     style="width:64px" maxlength="4" value="{_year}" autocomplete="off">
      </div>
      <div class="fw"><span class="fl">סמסטר</span>
        <input class="ti" id="f-semester" style="width:54px" maxlength="3" value="{_sem_disp}" autocomplete="off">
      </div>
      <div class="dash">—</div>
      <div class="fw">
        <span class="fl">(מחלקה . תואר . קורס)</span>
        <div style="display:flex;align-items:flex-end;gap:4px">
          <input class="ti" id="f-dept"   style="width:64px" maxlength="5" inputmode="numeric" value="{_dept}" autocomplete="off">
          <div class="sep">.</div>
          <input class="ti" id="f-degree" style="width:40px" maxlength="2" inputmode="numeric" value="{_deg}"  autocomplete="off">
          <div class="sep">.</div>
          <input class="ti" id="f-course" style="width:52px" maxlength="6" inputmode="numeric" value="{_cnum}" autocomplete="off">
        </div>
      </div>
      <div class="dash">—</div>
      <div class="fw"><span class="fl">מספר קורס</span>
        <input class="ti" id="f-num" style="width:64px" maxlength="6" inputmode="numeric" value="{_cnum}" autocomplete="off">
      </div>
    </div>
  </div>
  <div class="err" id="err-msg"></div>
  <div class="ctrl">
    <button id="tog" onclick="toggle()">⏹ עצור</button>
    <div class="stat">
      <div class="dot typing" id="sdot"></div>
      <span id="stxt">מקליד...</span>
    </div>
  </div>
  <div class="cyc-bar">
    <div class="cyc-track">
      <div class="cyc-fill" id="cyc-fill"></div>
      <div class="cyc-thumb" id="cyc-thumb"></div>
      <div class="cyc-start" title="תחילת מחזור">
        <div class="cyc-start-tip">▶ התחלה</div>
      </div>
    </div>
    <div class="cyc-lbl"><span style="color:#e53935;font-weight:700">|</span><span>◷ המתנה</span><span>↺ סיום</span></div>
  </div>
</div>
<script>
const CHAR_MS=100, PAUSE_MS=10000, CLEAR_MS=400;
const TYPE_MS=(3+1+4)*CHAR_MS+2*80+400;
const TOTAL_MS=TYPE_MS+PAUSE_MS+CLEAR_MS;
var cycStartTime=null;
(function rafLoop(){{
  if(cycStartTime!==null&&running){{
    var p=Math.min(99,(Date.now()-cycStartTime)/TOTAL_MS*100);
    var fill=document.getElementById('cyc-fill');
    var thumb=document.getElementById('cyc-thumb');
    if(fill)fill.style.width=p+'%';
    if(thumb)thumb.style.left=p+'%';
  }}
  requestAnimationFrame(rafLoop);
}})();
const FIELDS=[
  {{id:'f-dept',    val:'361'}},
  {{id:'f-degree',  val:'1'}},
  {{id:'f-course',  val:'3581'}},
];

const owned=FIELDS.map(()=>{_pre});
let running=true, timers=[];
const sch=(fn,ms)=>timers.push(setTimeout(fn,ms));
const cancAll=()=>{{timers.forEach(clearTimeout);timers=[];}};

FIELDS.forEach((f,i)=>{{
  const el=document.getElementById(f.id);
  el.addEventListener('focus',()=>{{if(!owned[i]){{el.value='';owned[i]=true;}}}});
  el.addEventListener('blur', ()=>{{if(!el.value.trim()) owned[i]=false;}});
  el.addEventListener('input',()=>{{owned[i]=true;}});
}});

document.getElementById('f-num').addEventListener('input',function(){{
  document.getElementById('f-course').value=this.value;
}});
document.getElementById('f-course').addEventListener('input',function(){{
  document.getElementById('f-num').value=this.value;
}});

document.getElementById('btn-add').onclick=function(){{
  const d=document.getElementById('f-dept').value.trim();
  const g=document.getElementById('f-degree').value.trim();
  const c=document.getElementById('f-course').value.trim();
  const yr=document.getElementById('f-year').value.trim();
  const sm_t=document.getElementById('f-semester').value.trim();
  const err=document.getElementById('err-msg');
  if(!d||!g||!c||!yr){{err.textContent='נא למלא את כל השדות';return;}}
  if(!/^[0-9]+$/.test(d)||!/^[0-9]+$/.test(g)||!/^[0-9]+$/.test(c)){{
    err.textContent='מחלקה, תואר וקורס חייבים להכיל ספרות בלבד';return;
  }}
  err.textContent='';
  const smMap={{"א'":"1","ב'":"2","קיץ":"3"}};
  const sm=smMap[sm_t]||"1";
  window.parent.location.search='?dept='+d+'&deg='+g+'&cnum='+c+'&yr='+yr+'&sem='+sm+'&sid='+Date.now();
}};

function setStatus(m){{
  document.getElementById('sdot').className='dot '+m;
  document.getElementById('stxt').textContent=m==='typing'?'מקליד...':m==='waiting'?'ממתין...':'עצור';
}}
function clrFields(){{
  FIELDS.forEach((f,i)=>{{if(!owned[i]){{const e=document.getElementById(f.id);e.value='';e.classList.remove('active');}}}});
}}
function typeField(fi,ci,done){{
  if(!running)return;
  const f=FIELDS[fi];
  if(owned[fi]){{sch(done,50);return;}}
  const el=document.getElementById(f.id);
  if(ci===0)el.classList.add('active');
  if(ci<=f.val.length){{
    el.value=f.val.slice(0,ci);
    sch(()=>typeField(fi,ci+1,done),CHAR_MS);
  }}else{{
    el.classList.remove('active');
    sch(done,80);
  }}
}}
function typeSeq(i,done){{
  if(!running)return;
  if(i>=FIELDS.length){{done();return;}}
  typeField(i,0,()=>typeSeq(i+1,done));
}}
function cycle(){{
  if(!running)return;
  cycStartTime=Date.now();
  var fill=document.getElementById('cyc-fill');
  var thumb=document.getElementById('cyc-thumb');
  if(fill)fill.style.width='0%';
  if(thumb)thumb.style.left='0%';
  setStatus('typing');
  typeSeq(0,()=>{{
    if(!running)return;
    setStatus('waiting');
    sch(()=>{{if(!running)return;clrFields();sch(cycle,CLEAR_MS);}},PAUSE_MS);
  }});
}}
function toggle(){{
  const b=document.getElementById('tog');
  if(running){{running=false;cancAll();setStatus('stopped');b.textContent='▶ הפעל';cycStartTime=null;var f=document.getElementById('cyc-fill'),t=document.getElementById('cyc-thumb');if(f)f.style.width='0%';if(t)t.style.left='0%';}}
  else{{running=true;b.textContent='⏹ עצור';clrFields();sch(cycle,200);}}
}}
cycle();
</script></body></html>""", height=345)

        if _sid and _sid != st.session_state.get("_tt_last_sid", "") and _dept and _deg and _cnum:
            st.session_state["_tt_last_sid"] = _sid
            with st.spinner("מחפש מערכת שעות ברקע..."):
                try:
                    result = _scrape_timetable_visible(_dept, _deg, _cnum, _year, _sem)
                    st.session_state.tt_result = result
                except Exception as exc:
                    st.error(f"שגיאה: {exc}")
                    st.session_state.tt_result = None

        result = st.session_state.get("tt_result")
        if result:
            if "error" in result:
                st.warning(result["error"])
            else:
                st.success(f"קורס: **{result.get('course_name', '')}**")
                sched = result.get("schedule", [])
                if not sched:
                    st.info("לא נמצאו נתוני מערכת שעות בדף")
                else:
                    for i, s in enumerate(sched, 1):
                        header = f"קבוצה {i}"
                        if s.get("type"):     header += f" – {s['type']}"
                        if s.get("lecturer"): header += f" | {s['lecturer']}"
                        with st.expander(header, expanded=True):
                            if s.get("times_raw"):
                                st.write(f"**זמן:** {s['times_raw']}")
                            if s.get("location"):
                                st.write(f"**מקום:** {s['location']}")
                            if s.get("method"):
                                st.write(f"**אופן:** {s['method']}")


    with tab_portal:
        st.subheader("🔐 כניסה לפורטל BGU")

        for _k, _v in [("portal_success",    None),
                        ("portal_usr_filled", False),
                        ("portal_pwd_filled", False),
                        ("portal_id_filled",  False)]:
            if _k not in st.session_state:
                st.session_state[_k] = _v

        def _icon_html(filled: bool, success=None) -> str:
            if success is True:  return "<span style='color:green;font-size:22px'>✔</span>"
            if success is False: return "<span style='color:red;font-size:22px'>✗</span>"
            if filled:           return "<span style='font-size:22px'>←</span>"
            return ""

        def _show(filled: bool, success=None):
            html = _icon_html(filled, success)
            if html:
                st.markdown(
                    f"<div style='margin-top:32px;text-align:center'>{html}</div>",
                    unsafe_allow_html=True)

        with st.form("portal_login_form"):
            c_usr, c_u_ic, c_pwd, c_p_ic, c_id, c_id_ic = st.columns(
                [0.28, 0.05, 0.27, 0.05, 0.27, 0.08])

            with c_usr:
                usr_val = st.text_input("שם משתמש")
            with c_u_ic:
                _show(st.session_state.portal_usr_filled)

            with c_pwd:
                pwd_val = st.text_input("סיסמה", type="password")
            with c_p_ic:
                _show(st.session_state.portal_pwd_filled)

            with c_id:
                id_val = st.text_input('מספר ת"ז')
            with c_id_ic:
                _show(st.session_state.portal_id_filled, st.session_state.portal_success)

            submitted = st.form_submit_button("כניסה לפורטל", type="primary",
                                               use_container_width=True)

        if submitted:
            st.session_state.portal_usr_filled = bool(usr_val.strip())
            st.session_state.portal_pwd_filled = bool(pwd_val.strip())
            st.session_state.portal_id_filled  = bool(id_val.strip())

            if not usr_val.strip() or not pwd_val.strip() or not id_val.strip():
                st.warning("נא למלא את כל השדות")
                st.session_state.portal_success = None
            else:
                with st.spinner("מתחבר לפורטל BGU…"):
                    try:
                        from PORTAL import login as _portal_login
                        st.session_state.portal_success = _portal_login(
                            usr_val.strip(), pwd_val.strip(), id_val.strip())
                    except ImportError:
                        st.error("שגיאה: הקובץ PORTAL.py חסר בסביבת השרת.")
                    except Exception as _exc:
                        st.session_state.portal_success = False
                        st.error(f"שגיאה בכניסה לפורטל: {_exc}")
            st.rerun()


# ── נקודת כניסה ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_ui()
