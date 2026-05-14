"""
ALL_CLAUDE.py
=============
בונה ChromaDB מקיף מכל קבצי ה-CSV בתיקיית data/.
הרץ פעם אחת (או בכל עדכון נתונים):
    .venv\\Scripts\\python.exe ALL_CLAUDE.py
"""

import json
import shutil
import time
from pathlib import Path

import chromadb
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db_storage"
COLLECTION = "bgu_knowledge"


# ── עזרים ────────────────────────────────────────────────────────────────────

def _val(row, *keys) -> str:
    for k in keys:
        v = row.get(k, "")
        if pd.notna(v) and str(v).strip() not in ("", "nan", "NaN"):
            return str(v).strip()
    return ""


def _load(fname: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{fname}.csv"
    if not path.exists():
        return None
    for enc in ("utf-8-sig", "cp1255", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None


# ── קבלה ──────────────────────────────────────────────────────────────────────

def admission_docs() -> list[dict]:
    docs = []
    for fname in ("bgu_admission", "bgu_admission_requirements", "bgu_admission_complete"):
        df = _load(fname)
        if df is None:
            continue
        for i, row in df.iterrows():
            name = _val(row, "שם_המסלול")
            if not name:
                continue
            skhum_hnd = _val(row, "סכם_הנדסה")
            skhum = _val(row, "סכם")
            skhum_kmt = _val(row, "סכם_כמותי")
            if skhum_hnd:
                entry = f"ציון כניסה (סכם הנדסה): {skhum_hnd}"
            elif skhum:
                entry = f"ציון כניסה (סכם): {skhum}"
            elif skhum_kmt:
                entry = f"ציון כניסה (סכם כמותי): {skhum_kmt}"
            else:
                entry = "ציון כניסה: לא נמצא"

            lines = [
                f"מסלול: {name}",
                f"סוג קבלה: {_val(row, 'סוג_קבלה')}",
                entry,
            ]
            for lbl, key in [
                ("פסיכומטרי מינימום", "פסיכומטרי"),
                ("ממוצע בגרות", "ממוצע_בגרות"),
                ("מתמטיקה", "מתמטיקה"),
                ("פיסיקה", "פיסיקה"),
                ("אנגלית", "אנגלית"),
                ("עברית", "עברית"),
            ]:
                v = _val(row, key)
                if v and len(v) < 60:
                    lines.append(f"{lbl}: {v}")
            raw = _val(row, "תוכן_גולמי")
            if raw and len(raw) < 800:
                lines.append(f"מידע נוסף: {raw[:800]}")
            url = _val(row, "URL")
            if url:
                lines.append(f"קישור לדף תנאי הקבלה: {url}")

            docs.append({
                "id": f"admission_{fname}_{i}",
                "text": "\n".join(lines),
                "meta": {"type": "admission", "source": fname, "program": name, "url": url},
            })
    return docs


# ── ציונים ────────────────────────────────────────────────────────────────────

def grades_docs() -> list[dict]:
    docs = []
    seen: set[str] = set()
    GRADE_FILES = [
        "g_all_courses_with_grades",
        "grades",
        "grades_machinery",
        "machinery2_with_grades",
        "management_grades_extracted",
    ]
    for fname in GRADE_FILES:
        df = _load(fname)
        if df is None:
            continue
        sem_cols = [c for c in df.columns if any(y in str(c) for y in ("2022", "2023", "2024", "2025"))]
        for _, row in df.iterrows():
            name = _val(row, "שם קורס", "Course_Name")
            num = _val(row, "מספר קורס", "Course_Number")
            if not name or num in seen:
                continue
            seen.add(num)
            lines = [
                f"קורס: {name}",
                f"מספר קורס: {num}",
                f"נקודות זכות: {_val(row, 'נק\"ז', 'Credits')}",
            ]
            prereq = _val(row, "קורסי חובת מעבר", "Prerequisites")
            if prereq:
                lines.append(f"קורסי קדם: {prereq}")
            for sem in sem_cols:
                v = _val(row, sem)
                if v and v != "Label Not Found":
                    sem_label = str(sem).replace("A", " א'").replace("B", " ב'")
                    lines.append(f"ציון {sem_label}: {v}")
            docs.append({
                "id": f"grades_{fname}_{num}",
                "text": "\n".join(lines),
                "meta": {"type": "grades", "source": fname, "course_name": name, "course_num": num},
            })
    return docs


# ── מלגות ─────────────────────────────────────────────────────────────────────

def scholarship_docs() -> list[dict]:
    docs = []
    for fname in ("bgu_scholarships_new_3", "bgu_scholarships_deep_scan", "bgu_scholarships_progress"):
        df = _load(fname)
        if df is None:
            continue
        for i, row in df.iterrows():
            name = _val(row, "שם המלגה", "name")
            if not name:
                continue
            lines = [f"מלגה: {name}"]
            raw = _val(row, "תוכן_נקי_JSON", "content")
            if raw:
                try:
                    data = json.loads(raw)
                    for k, v in data.items():
                        if v and str(v) not in ("nan", ""):
                            lines.append(f"{k}: {str(v)[:300]}")
                except Exception:
                    lines.append(f"תוכן: {raw[:600]}")
            url = _val(row, "URL")
            if url:
                lines.append(f"קישור ישיר למלגה: {url}")
            docs.append({
                "id": f"scholarship_{fname}_{i}",
                "text": "\n".join(lines),
                "meta": {"type": "scholarship", "source": fname, "name": name, "url": url},
            })
    return docs


# ── פרויקטים ─────────────────────────────────────────────────────────────────

def project_docs() -> list[dict]:
    docs = []
    for fname in ("Projects_Classified", "Projects_website", "PROJECT_WEB1"):
        df = _load(fname)
        if df is None:
            continue
        for i, row in df.iterrows():
            name = _val(row, "Project Name", "שם_פרויקט", "Title")
            if not name:
                continue
            lines = [f"פרויקט: {name}"]
            for lbl, key in [
                ("פקולטה", "Faculty"),
                ("מחלקה", "Department"),
                ("יועץ", "Advisers"),
                ("מילות מפתח", "Keywords"),
                ("שנה", "Year"),
                ("התמחות", "Specialization"),
                ("תיאור", "Description"),
                ("נושא", "Topic"),
            ]:
                v = _val(row, key)
                if v:
                    lines.append(f"{lbl}: {v[:300]}")
            docs.append({
                "id": f"project_{fname}_{i}",
                "text": "\n".join(lines),
                "meta": {"type": "project", "source": fname, "project_name": name},
            })
    return docs


# ── קורסים ───────────────────────────────────────────────────────────────────

def course_docs() -> list[dict]:
    docs = []
    seen: set[str] = set()
    for fname in ("g_all_courses", "courses_fixed", "machinery", "machinery2", "management_courses_fixed"):
        df = _load(fname)
        if df is None:
            continue
        for _, row in df.iterrows():
            name = _val(row, "שם קורס", "Course_Name")
            num = _val(row, "מספר קורס", "Course_Number")
            if not name or num in seen:
                continue
            seen.add(num)
            lines = [
                f"קורס: {name}",
                f"מספר קורס: {num}",
                f"נקודות זכות: {_val(row, 'נק\"ז', 'Credits')}",
            ]
            prereq = _val(row, "קורסי חובת מעבר", "Prerequisites")
            if prereq:
                lines.append(f"קורסי קדם: {prereq}")
            lines.append(f"מקור: {fname}")
            docs.append({
                "id": f"course_{fname}_{num}",
                "text": "\n".join(lines),
                "meta": {"type": "course", "source": fname, "course_name": name},
            })
    return docs


# ── צבא ──────────────────────────────────────────────────────────────────────

def army_docs() -> list[dict]:
    df = _load("army_results")
    if df is None:
        return []
    docs = []
    for i, row in df.iterrows():
        title = _val(row, "title")
        text = _val(row, "text")
        topic = _val(row, "topic")
        if not text:
            continue
        docs.append({
            "id": f"army_{i}",
            "text": f"נושא: {topic}\nכותרת: {title}\nתוכן: {text[:1000]}",
            "meta": {"type": "army", "source": "army_results", "title": title},
        })
    return docs


def partner_docs() -> list[dict]:
    """ידע כללי מאתר BGU – דפים, לינקים ונושאים (bgu_partner_knowledge.csv)."""
    df = _load("bgu_partner_knowledge")
    if df is None:
        return []
    docs = []
    for i, row in df.iterrows():
        title   = _val(row, "title")
        summary = _val(row, "summary", "micro_summary")
        if not title and not summary:
            continue
        lines = []
        if title:
            lines.append(f"כותרת: {title}")
        micro = _val(row, "micro_summary")
        if micro and micro != title:
            lines.append(f"תקציר: {micro}")
        if summary and summary != micro:
            lines.append(f"תוכן: {summary[:300]}")
        tags = _val(row, "tags")
        if tags:
            lines.append(f"נושאים: {tags}")
        url = _val(row, "url")
        if url:
            lines.append(f"קישור ישיר: {url}")
        doc_type = _val(row, "type") or "partner"
        docs.append({
            "id": f"partner_{i}",
            "text": "\n".join(lines),
            "meta": {"type": doc_type, "source": "bgu_partner_knowledge",
                     "title": title, "url": url},
        })
    return docs


def people_docs() -> list[dict]:
    df = _load("people_of_bgu")
    if df is None:
        return []
    docs = []
    for i, row in df.iterrows():
        name = _val(row, "name")
        if not name:
            continue
        email   = _val(row, "email")
        faculty = _val(row, "faculty")
        lines = [f"שם: {name}"]
        if email:
            lines.append(f"מייל: {email}")
        if faculty:
            lines.append(f"פקולטה/יחידה: {faculty}")
        docs.append({
            "id": f"person_{i}",
            "text": "\n".join(lines),
            "meta": {"type": "people", "source": "people_of_bgu", "name": name},
        })
    return docs


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("ALL_CLAUDE – בונה ChromaDB מקיף מכל קבצי CSV...\n")

    # מחיקה מלאה של תיקיית ChromaDB כדי למנוע קורפשן מריצות קודמות
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        if CHROMA_DIR.exists():
            print("  אזהרה: לא ניתן למחוק תיקייה ישנה (קובץ נעול) – ממשיך בכל זאת")
        else:
            print("  נמחקה תיקיית chroma_db_storage הישנה")
    CHROMA_DIR.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # מחיקת collection אם קיים (אחרי rmtree זה בדרך כלל לא קיים, אבל ליתר ביטחון)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    col = client.create_collection(COLLECTION)
    time.sleep(1)  # allow Rust backend to flush collection metadata
    col = client.get_collection(COLLECTION)  # re-fetch for stable reference
    print(f"  נוצר collection: {COLLECTION}\n")

    builders = [
        ("קבלה",       admission_docs),
        ("ציונים",     grades_docs),
        ("מלגות",      scholarship_docs),
        ("פרויקטים",   project_docs),
        ("קורסים",     course_docs),
        ("צבא",        army_docs),
    ]

    total = 0
    for label, builder in builders:
        docs = builder()
        if not docs:
            print(f"  {label}: לא נמצאו מסמכים")
            continue

        seen_ids: set[str] = set()
        unique = []
        for d in docs:
            if d["id"] not in seen_ids:
                seen_ids.add(d["id"])
                unique.append(d)

        # batch 25 – upsert is more stable than add on Rust backend (Windows)
        for start in range(0, len(unique), 25):
            chunk = unique[start : start + 25]
            for attempt in range(3):
                try:
                    col.upsert(
                        documents=[d["text"] for d in chunk],
                        ids=[d["id"] for d in chunk],
                        metadatas=[d["meta"] for d in chunk],
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(1)
                    col = client.get_or_create_collection(COLLECTION)

        print(f"  {label}: {len(unique)} מסמכים נוספו")
        total += len(unique)

    print(f'\n✓ סה"כ {total} מסמכים ב-ChromaDB.')
    print("  הרץ BGUManager.py להתחיל שיחה.")


if __name__ == "__main__":
    main()
