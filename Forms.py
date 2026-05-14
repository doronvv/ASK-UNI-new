# -*- coding: utf-8 -*-
"""
BGU PDF Crawler — Forms.py
כל הרצה סורקת עד MAX_PAGES_PER_RUN דפים ושומרת מצב לקובץ JSON.
הרצה הבאה ממשיכה מהמקום שנעצרנו.
"""
import os
import re
import time
import json
import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

# ── הגדרות ──────────────────────────────────────────────────────────────────
DOWNLOAD_DIR  = r"C:\Users\doron\PycharmProjects\PythonProject3\.venv\downloads\טפסים"
STATE_FILE    = r"C:\Users\doron\PycharmProjects\PythonProject3\crawl_state.json"
LOG_FILE      = r"C:\Users\doron\PycharmProjects\PythonProject3\crawl_log.txt"

ALLOWED_DOMAIN   = "bgu.ac.il"
MAX_PAGES_PER_RUN = 200
PAGE_TIMEOUT     = 15
RETRY_WAIT       = 5
MAX_RETRIES      = 2
POLITE_DELAY     = 0.4   # שניות בין בקשות

# כתובות זרע — נקודות פתיחה ידועות באתר BGU
SEED_URLS = [
    "https://www.bgu.ac.il/",
    "https://www.bgu.ac.il/sitemap.xml",
    "https://in.bgu.ac.il/",
    "https://in.bgu.ac.il/sitemap.aspx",
    "https://www.bgu.ac.il/welcome/",
    "https://www.bgu.ac.il/u/",
    "https://www.bgu.ac.il/u/academic-affairs/",
    "https://www.bgu.ac.il/u/academic-affairs/dekanat/",
    "https://www.bgu.ac.il/u/academic-affairs/dekanat/miluim/",
    "https://www.bgu.ac.il/welcome/contents/course-registration/",
    "https://www.bgu.ac.il/welcome/contents/course-registration/info/",
    "https://www.bgu.ac.il/u/student-affairs/",
    "https://www.bgu.ac.il/u/finance/",
    "https://www.bgu.ac.il/u/hr/",
    "https://www.bgu.ac.il/u/library/",
    "https://www.bgu.ac.il/u/safety/",
    "https://in.bgu.ac.il/Pages/default.aspx",
    "https://in.bgu.ac.il/bashanah/Pages/default.aspx",
    "https://in.bgu.ac.il/bgu/hr/Pages/default.aspx",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "he,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
}
# ────────────────────────────────────────────────────────────────────────────


def setup():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ── עזרי URL ────────────────────────────────────────────────────────────────
def is_same_domain(url: str) -> bool:
    try:
        return ALLOWED_DOMAIN in urlparse(url).netloc
    except Exception:
        return False


def is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def normalize(url: str) -> str:
    try:
        p = urlparse(url)
        return p._replace(fragment="", query="").geturl().rstrip("/")
    except Exception:
        return url.rstrip("/")


def sanitize_filename(url: str) -> str:
    name = urlparse(url).path.split("/")[-1] or "document.pdf"
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def unique_filepath(filename: str) -> str:
    path = os.path.join(DOWNLOAD_DIR, filename)
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(filename)
    i = 1
    while os.path.exists(os.path.join(DOWNLOAD_DIR, f"{base}_{i}{ext}")):
        i += 1
    return os.path.join(DOWNLOAD_DIR, f"{base}_{i}{ext}")


# ── שמירה / טעינה של מצב ────────────────────────────────────────────────────
def load_state() -> tuple[set, deque, set]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            visited        = set(data.get("visited", []))
            queue          = deque(data.get("queue", []))
            downloaded_pdfs = set(data.get("downloaded_pdfs", []))
            print(f"[STATE] Loaded: {len(visited)} visited, {len(queue)} in queue, {len(downloaded_pdfs)} PDFs")
            return visited, queue, downloaded_pdfs
        except Exception as e:
            print(f"[STATE] Failed to load state ({e}), starting fresh")

    # התחלה מחדש — טעינת כתובות זרע
    seeds = [normalize(u) for u in SEED_URLS]
    return set(), deque(seeds), set()


def save_state(visited: set, queue: deque, downloaded_pdfs: set):
    data = {
        "visited":         list(visited),
        "queue":           list(queue),
        "downloaded_pdfs": list(downloaded_pdfs),
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def log(msg: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ── רשת ─────────────────────────────────────────────────────────────────────
def fetch(url: str, session: requests.Session, stream: bool = False):
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=PAGE_TIMEOUT, stream=stream, allow_redirects=True)
            if resp.status_code == 200:
                return resp
            print(f"  HTTP {resp.status_code}: {url}")
            return None
        except requests.exceptions.Timeout:
            print(f"  Timeout ({attempt+1}/{MAX_RETRIES}): {url}")
        except requests.exceptions.ConnectionError:
            print(f"  Connection error ({attempt+1}/{MAX_RETRIES}): {url}")
        except Exception as e:
            print(f"  Error ({attempt+1}/{MAX_RETRIES}): {e}")
        if attempt < MAX_RETRIES - 1:
            print(f"  Waiting {RETRY_WAIT}s...")
            time.sleep(RETRY_WAIT)
    return None


def download_pdf(url: str, session: requests.Session) -> bool:
    resp = fetch(url, session, stream=True)
    if resp is None:
        return False
    ct = resp.headers.get("content-type", "").lower()
    if "pdf" not in ct and not is_pdf_url(url):
        return False
    filename = sanitize_filename(url)
    filepath = unique_filepath(filename)
    try:
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        msg = f"  [PDF saved] {os.path.basename(filepath)}"
        print(msg)
        log(f"PDF: {url} -> {os.path.basename(filepath)}")
        return True
    except Exception as e:
        print(f"  Save failed: {e}")
        return False


# ── חילוץ קישורים ────────────────────────────────────────────────────────────
def extract_links_html(content: bytes, base_url: str) -> list[str]:
    try:
        soup = BeautifulSoup(content, "html.parser")
    except Exception:
        return []
    results = []
    # כל תגיות עם href
    for tag in soup.find_all(True, href=True):
        href = tag.get("href") or ""
        if isinstance(href, list):
            href = href[0] if href else ""
        href = str(href).strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        full = urljoin(base_url, href)
        if urlparse(full).scheme in ("http", "https"):
            results.append(normalize(full))
    # src — iframe / embed
    for tag in soup.find_all(["iframe", "embed", "object", "source"], src=True):
        src = tag.get("src") or ""
        if isinstance(src, list):
            src = src[0] if src else ""
        src = str(src).strip()
        if src:
            full = urljoin(base_url, src)
            if urlparse(full).scheme in ("http", "https"):
                results.append(normalize(full))
    # חיפוש כתובות PDF ישירות בטקסט (לאתרים שמסתירים קישורים ב-JS)
    text = content.decode("utf-8", errors="ignore")
    for match in re.findall(r'["\']([^"\']*\.pdf)["\']', text, re.IGNORECASE):
        full = urljoin(base_url, match)
        if urlparse(full).scheme in ("http", "https"):
            results.append(normalize(full))
    return results


def extract_links_sitemap(content: bytes, base_url: str) -> list[str]:
    results = []
    try:
        root = ET.fromstring(content)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for loc in root.findall(".//sm:loc", ns):
            if loc.text:
                results.append(normalize(loc.text.strip()))
        # sitemapindex
        for loc in root.findall(".//sm:sitemap/sm:loc", ns):
            if loc.text:
                results.append(normalize(loc.text.strip()))
    except Exception:
        pass
    return results


# ── לולאת סריקה ─────────────────────────────────────────────────────────────
def crawl():
    setup()
    visited, queue, downloaded_pdfs = load_state()
    queued_set = set(queue) | visited   # למניעת כפילויות ב-queue

    # הוסף seeds חדשים שעדיין לא ביקרנו בהם
    for s in [normalize(u) for u in SEED_URLS]:
        if s not in queued_set:
            queue.append(s)
            queued_set.add(s)

    session = requests.Session()
    session.headers.update(HEADERS)

    pages_this_run = 0
    pdf_this_run   = 0

    print(f"\n{'='*70}")
    print(f"BGU PDF Crawler — הרצה חדשה")
    print(f"בתור: {len(queue)} דפים | נסרקו עד כה: {len(visited)} | PDFים: {len(downloaded_pdfs)}")
    print(f"מגבלה להרצה זו: {MAX_PAGES_PER_RUN} דפים")
    print(f"{'='*70}\n")

    autosave_every = 50  # שמור מצב כל כמה דפים

    def print_summary(reason: str):
        print(f"\n{'='*70}")
        print(f"הרצה הסתיימה ({reason}).")
        print(f"דפים שנסרקו בהרצה זו : {pages_this_run}")
        print(f"PDFים שהורדו בהרצה זו : {pdf_this_run}")
        print(f"סה\"כ דפים שנסרקו      : {len(visited)}")
        print(f"סה\"כ PDFים שהורדו      : {len(downloaded_pdfs)}")
        print(f"נשארו בתור             : {len(queue)} דפים")
        if queue:
            print(f"הרץ שוב את הסקריפט כדי להמשיך מהמקום שנעצרת.")
        else:
            print(f"הסריקה הושלמה במלואה — אין יותר דפים לבדוק.")
        print(f"{'='*70}\n")

    try:
        while queue and pages_this_run < MAX_PAGES_PER_RUN:
            url = queue.popleft()

            if url in visited:
                continue
            visited.add(url)
            pages_this_run += 1

            # שמירה אוטומטית כל AUTOSAVE_EVERY דפים
            if pages_this_run % autosave_every == 0:
                save_state(visited, queue, downloaded_pdfs)
                print(f"  [שמירה אוטומטית — {pages_this_run} דפים]")

            # --- PDF ישיר ---
            if is_pdf_url(url):
                if url not in downloaded_pdfs:
                    print(f"[PDF] {url}")
                    if download_pdf(url, session):
                        downloaded_pdfs.add(url)
                        pdf_this_run += 1
                continue

            print(f"[{pages_this_run}/{MAX_PAGES_PER_RUN}] {url}")

            resp = fetch(url, session)
            if resp is None:
                log(f"SKIP (no response): {url}")
                continue

            ct = resp.headers.get("content-type", "").lower()
            final_url = normalize(resp.url)  # לאחר redirect

            # PDF שהגיע ללא סיומת .pdf
            if "pdf" in ct:
                if url not in downloaded_pdfs:
                    filename = sanitize_filename(url)
                    fp = unique_filepath(filename)
                    try:
                        with open(fp, "wb") as f:
                            f.write(resp.content)
                        print(f"  [PDF saved] {os.path.basename(fp)}")
                        log(f"PDF (content-type): {url}")
                        downloaded_pdfs.add(url)
                        pdf_this_run += 1
                    except Exception as e:
                        print(f"  Save error: {e}")
                continue

            # Sitemap XML
            if "xml" in ct or url.endswith(".xml"):
                links = extract_links_sitemap(resp.content, final_url)
            elif "html" in ct:
                links = extract_links_html(resp.content, final_url)
            else:
                continue

            new_pdfs = 0
            new_pages = 0
            for link in links:
                if not is_same_domain(link):
                    continue
                if link in queued_set:
                    continue
                queued_set.add(link)
                if is_pdf_url(link):
                    queue.appendleft(link)   # PDFs קודמים
                    new_pdfs += 1
                else:
                    queue.append(link)
                    new_pages += 1

            if new_pdfs or new_pages:
                print(f"  +{new_pdfs} PDF  +{new_pages} pages  (queue={len(queue)})")

            time.sleep(POLITE_DELAY)

    except KeyboardInterrupt:
        print("\n\n[!] נעצר ידנית — שומר מצב...")
        save_state(visited, queue, downloaded_pdfs)
        print_summary("נעצר ידנית")
        return

    # סיום תקין
    save_state(visited, queue, downloaded_pdfs)
    print_summary("הגיע למגבלת הדפים" if queue else "סיום מלא")


if __name__ == "__main__":
    crawl()
