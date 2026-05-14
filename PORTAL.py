"""
PORTAL.py – כניסה לפורטל הסטודנטים של BGU
==========================================
הרצה:  python PORTAL.py
דרישות: pip install undetected-chromedriver selenium
"""

import sys
import time
import random
from pathlib import Path

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By

PORTAL_URL = "https://portal.bgu.ac.il/public/login?returnUrl=private%2Fhome"
DEBUG_DIR  = Path(__file__).parent / "debug_pages"
DEBUG_DIR.mkdir(exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def human_sleep(a: float = 0.5, b: float = 1.5):
    """השהייה אקראית בין a ל-b שניות."""
    time.sleep(random.uniform(a, b))


def human_type(el, text: str, lo: float = 0.05, hi: float = 0.22):
    """הקלדה תו-תו עם השהייה אקראית – מחקה הקלדה אנושית."""
    for ch in text:
        el.send_keys(ch)
        time.sleep(random.uniform(lo, hi))


def make_driver():
    opts = uc.ChromeOptions()
    opts.add_argument("--start-maximized")
    opts.add_argument(f"--user-agent={USER_AGENT}")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    driver = uc.Chrome(options=opts, use_subprocess=True)
    # הסתר עוד סימני אוטומציה
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    return driver


def angular_fill(driver, el, text: str, lo: float = 0.05, hi: float = 0.22):
    """מלא שדה Angular – הקלדה אנושית + שליחת events לזיהוי Angular."""
    driver.execute_script("arguments[0].focus();", el)
    el.clear()
    human_type(el, text, lo=lo, hi=hi)
    driver.execute_script("""
        var el = arguments[0];
        ['input','change','blur'].forEach(function(ev){
            el.dispatchEvent(new Event(ev, {bubbles:true}));
        });
    """, el)


def wait_visible(driver, css, timeout=15):
    """המתן עד שהאלמנט יופיע ויהיה נראה."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            els = driver.find_elements(By.CSS_SELECTOR, css)
            for el in els:
                if el.is_displayed() and el.is_enabled():
                    return el
        except Exception:
            pass
        time.sleep(0.5)
    return None


def login(username: str, password: str, id_num: str) -> bool:
    driver = make_driver()
    success = False
    try:
        print("\n  🌐  פותח פורטל BGU…")
        driver.get(PORTAL_URL)
        human_sleep(3, 4)  # המתנה כמו אדם שמחכה לטעינה

        (DEBUG_DIR / "portal_login.html").write_text(
            driver.page_source, encoding="utf-8")

        # ── מצא שדות Angular ─────────────────────────────────────────────
        user_el = wait_visible(driver, "input[formcontrolname='username']")
        pass_el = wait_visible(driver, "input[formcontrolname='password']")
        id_el   = wait_visible(driver, "input[formcontrolname='id']")

        if not user_el:
            print("  ⚠  לא נמצא שדה שם משתמש")
            driver.save_screenshot(str(DEBUG_DIR / "portal_login.png"))
            return False

        # ── מלא שם משתמש ──────────────────────────────────────────────────
        print("  📝  ממלא שם משתמש…")
        human_sleep(0.5, 0.9)
        angular_fill(driver, user_el, username, lo=0.02, hi=0.04)
        human_sleep(0.5, 1.0)

        # ── מלא סיסמה ─────────────────────────────────────────────────────
        if pass_el:
            print("  📝  ממלא סיסמה…")
            human_sleep(0.2, 0.5)
            angular_fill(driver, pass_el, password, lo=0.02, hi=0.05)
            human_sleep(0.6, 1.0)
        else:
            print("  ⚠  לא נמצא שדה סיסמה")

        # ── מלא תעודת זהות ────────────────────────────────────────────────
        if id_el:
            print("  📝  ממלא מספר תעודת זהות…")
            human_sleep(0.2, 0.5)
            angular_fill(driver, id_el, id_num, lo=0.03, hi=0.05)
            human_sleep(0.5, 0.7)
        else:
            print("  ⚠  לא נמצא שדה ת\"ז")

        # ── לחץ כניסה ─────────────────────────────────────────────────────
        login_btn = wait_visible(driver, "button[mat-flat-button]")
        if not login_btn:
            for xp in [
                "//*[contains(text(),'התחברות')]",
                "//*[contains(text(),'להתחבר')]",
                "//button[contains(text(),'Login')]",
                "//button[@type='submit']",
            ]:
                try:
                    el = driver.find_element(By.XPATH, xp)
                    if el.is_displayed():
                        login_btn = el; break
                except Exception:
                    pass

        if login_btn:
            print("  🖱   לוחץ התחברות…")
            human_sleep(0.8, 1.5)  # השהייה לפני הלחיצה – כמו שאדם חושב רגע
            driver.execute_script("arguments[0].click();", login_btn)
        else:
            print("  ⚠  לא נמצא כפתור התחברות")
            return False

        # ── בדוק תוצאה ────────────────────────────────────────────────────
        human_sleep(2, 3)

        current_url = driver.current_url
        success = "public/login" not in current_url

        if success:
            print("  ✔  כניסה הצליחה!")
        else:
            print("  ✗  כניסה נכשלה.")

        return success

    finally:
        time.sleep(2)
        driver.quit()


def main():
    print("=" * 50)
    print("  BGU – כניסה לפורטל הסטודנטים")
    print("=" * 50)

    username = input("\n  שם משתמש        : ").strip()
    password = input("  סיסמה           : ").strip()
    id_num   = input("  מספר תעודת זהות : ").strip()

    if not username or not password or not id_num:
        print("\n  ❌  נא למלא את כל השדות.")
        sys.exit(1)

    try:
        success = login(username, password, id_num)
    except Exception as e:
        print(f"\n  ❌  שגיאה: {e}")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
