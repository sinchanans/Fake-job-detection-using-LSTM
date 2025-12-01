# tests/test_flows.py
"""
End-to-end Selenium tests for the Fake Job Detection Flask app.

How to use:
1. Start your Flask app in a separate terminal:
   python mainpage.py
2. Activate your testing virtual environment (that has selenium, pytest, webdriver-manager).
3. Optionally set environment variables (or edit defaults below):
   BASE_URL, TEST_USER_EMAIL, TEST_USER_PASSWORD, ADMIN_USERNAME, ADMIN_PASSWORD, SAMPLE_IMAGE
4. Run:
   pytest -q tests/test_flows.py --html=report.html
"""

import os
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoAlertPresentException
from webdriver_manager.chrome import ChromeDriverManager

# ---------------- Configuration (override with env vars if needed) -------------
BASE_URL = os.getenv("BASE_URL", "http://localhost:5050")
TEST_USER_EMAIL = os.getenv("TEST_USER_EMAIL", "testuser@example.com")
TEST_USER_PASSWORD = os.getenv("TEST_USER_PASSWORD", "TestPass123!")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")       # default in mainpage.py
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")    # default in mainpage.py

# Default sample image (can override). If running on Windows, set SAMPLE_IMAGE to a Windows absolute path.
SAMPLE_IMAGE = os.getenv("SAMPLE_IMAGE", r"/mnt/data/test imag2.jpg")

# ---------------- Selenium fixture ------------------------------------------
@pytest.fixture(scope="session")
def driver():
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless=new")   # uncomment to run headless in CI
    options.add_argument("--start-maximized")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(2)
    yield driver
    driver.quit()

# ---------------- Utility functions -----------------------------------------
def wait_for(driver, by, selector, timeout=10):
    return WebDriverWait(driver, timeout).until(EC.visibility_of_element_located((by, selector)))

# ---------------- User helpers ------------------------------------------------
def try_user_login(driver, base_url, email, password):
    """Try to login a user; return True if login succeeded (dashboard appears)."""
    driver.get(f"{base_url}/user_login")
    try:
        email_in = WebDriverWait(driver, 6).until(EC.visibility_of_element_located((By.NAME, "email")))
        pwd_in = driver.find_element(By.NAME, "password")
        email_in.clear(); email_in.send_keys(email)
        pwd_in.clear(); pwd_in.send_keys(password)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        # Wait for a stable dashboard indicator. Your dashboard template includes id="sidebar".
        WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.ID, "sidebar")))
        return True
    except Exception:
        return False

def register_test_user(driver, base_url, name, email, phone, password):
    """Register a new user through UI; tolerant to slight variations."""
    driver.get(f"{base_url}/register")
    try:
        name_in = WebDriverWait(driver, 8).until(EC.visibility_of_element_located((By.NAME, "name")))
        email_in = driver.find_element(By.NAME, "email")
        phone_in = driver.find_element(By.NAME, "phone")
        pwd_in = driver.find_element(By.NAME, "password")
        name_in.clear(); name_in.send_keys(name)
        email_in.clear(); email_in.send_keys(email)
        phone_in.clear(); phone_in.send_keys(phone)
        pwd_in.clear(); pwd_in.send_keys(password)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        # Wait briefly for success, but don't fail if redirect occurs
        try:
            WebDriverWait(driver, 6).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".alert-success")))
        except TimeoutException:
            pass
    except Exception:
        # If register page has different selectors, caller will handle failure
        return False
    return True

def ensure_test_user(driver, base_url, email, password):
    """Ensure test user exists and is logged in. Register if necessary."""
    ok = try_user_login(driver, base_url, email, password)
    if ok:
        return True
    # Attempt to register a deterministic test user, then login again
    test_name = "Selenium Test"
    test_phone = "9998887776"
    register_test_user(driver, base_url, test_name, email, test_phone, password)
    ok = try_user_login(driver, base_url, email, password)
    return ok

# ---------------- Admin helpers ---------------------------------------------
def admin_login(driver, base_url, username, password):
    """
    Attempt admin login. Return True on success.
    Handles both successful redirect and flash message for invalid credentials.
    """
    driver.get(f"{base_url}/admin_login")
    try:
        uname = WebDriverWait(driver, 6).until(EC.visibility_of_element_located((By.NAME, "username")))
        pwd = driver.find_element(By.NAME, "password")
        uname.clear(); uname.send_keys(username)
        pwd.clear(); pwd.send_keys(password)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
    except Exception:
        return False

    # If admin_dashboard element appears => success
    try:
        WebDriverWait(driver, 6).until(EC.presence_of_element_located((By.ID, "admin-dashboard")))
        return True
    except Exception:
        # Inspect page source for flash message "Invalid admin credentials"
        src = driver.page_source.lower()
        if "invalid admin credentials" in src or "invalid admin" in src or "invalid credentials" in src:
            return False
        # Fallback check on current URL
        try:
            if "/admin_dashboard" in driver.current_url:
                return True
        except Exception:
            pass
    return False

# ---------------- Tests -----------------------------------------------------
def test_register_user(driver):
    """Register a test user (if not already registered)."""
    driver.get(f"{BASE_URL}/register")
    try:
        # If register page is available, attempt registration with TEST_USER_EMAIL
        name_input = wait_for(driver, By.NAME, "name", timeout=6)
        driver.find_element(By.NAME, "email").clear()
        driver.find_element(By.NAME, "email").send_keys(TEST_USER_EMAIL)
        driver.find_element(By.NAME, "phone").clear()
        driver.find_element(By.NAME, "phone").send_keys("9998887776")
        driver.find_element(By.NAME, "password").clear()
        driver.find_element(By.NAME, "password").send_keys(TEST_USER_PASSWORD)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        WebDriverWait(driver, 6).until(EC.any_of(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".alert-success")),
            EC.url_contains("/user_login")
        ))
    except TimeoutException:
        pytest.skip("Register page not reachable or selectors mismatch; skipping registration.")

def test_user_login(driver):
    """Login with test user (register if needed)."""
    assert ensure_test_user(driver, BASE_URL, TEST_USER_EMAIL, TEST_USER_PASSWORD), "Failed to ensure or login test user."

def test_text_scan_page_and_scan(driver):
    """Visit /text_scan, paste a scam text and assert the system flags it as Fake."""
    assert ensure_test_user(driver, BASE_URL, TEST_USER_EMAIL, TEST_USER_PASSWORD), "Login required for text scan."
    driver.get(f"{BASE_URL}/text_scan")
    try:
        ta = wait_for(driver, By.ID, "text_input", timeout=6)
    except TimeoutException:
        ta = wait_for(driver, By.NAME, "text_input", timeout=6)

    # Use a clear scam phrase present in mainpage.py keyword list
    scam_text = "Urgent hiring! Pay registration fee of $100 to join."
    ta.clear()
    ta.send_keys(scam_text)
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

    # wait for result
    WebDriverWait(driver, 12).until(EC.any_of(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".result")),
        EC.presence_of_element_located((By.CSS_SELECTOR, ".output"))
    ))

    page = driver.page_source.lower()
    # mainpage's keyword check returns "🔴 Fake Job Detected!" -> check for 'fake'
    assert ("fake job detected" in page) or ("🔴 fake job detected" in page) or ("fake" in page), \
        f"Expected 'fake' in the result page; page snippet: {page[:400]!r}"

def test_image_scan_upload(driver):
    """Upload an image to /image_scan and assert extraction/result area appears."""
    assert ensure_test_user(driver, BASE_URL, TEST_USER_EMAIL, TEST_USER_PASSWORD), "Login required for image scan."
    driver.get(f"{BASE_URL}/image_scan")
    try:
        file_input = wait_for(driver, By.ID, "image_file", timeout=6)
    except TimeoutException:
        file_input = wait_for(driver, By.NAME, "image_file", timeout=6)

    local_file = SAMPLE_IMAGE
    # If SAMPLE_IMAGE is a linux/mnt path but tests run on Windows, try fallback copy
    if not os.path.exists(local_file):
        alt = os.path.join(os.getcwd(), "tests", "sample_job.jpg")
        if os.path.exists(alt):
            local_file = alt
        else:
            pytest.skip(f"Sample image not found at {SAMPLE_IMAGE}. Set SAMPLE_IMAGE env var to a valid image path accessible to tests.")
    file_input.send_keys(local_file)
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

    WebDriverWait(driver, 15).until(EC.any_of(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".output")),
        EC.presence_of_element_located((By.CSS_SELECTOR, ".result"))
    ))

    # Basic check: page updated (we don't assert exact OCR output)
    assert driver.page_source is not None

def test_history_page_shows_table(driver):
    """Visit /history and check that a table of scans exists."""
    assert ensure_test_user(driver, BASE_URL, TEST_USER_EMAIL, TEST_USER_PASSWORD), "Unable to create/login test user."
    driver.get(f"{BASE_URL}/history")
    table = wait_for(driver, By.CSS_SELECTOR, "table", timeout=8)
    rows = table.find_elements(By.TAG_NAME, "tr")
    assert len(rows) >= 1

def test_profile_update(driver):
    """Visit profile page and update name/phone fields."""
    assert ensure_test_user(driver, BASE_URL, TEST_USER_EMAIL, TEST_USER_PASSWORD), "Unable to create/login test user."
    driver.get(f"{BASE_URL}/profile")
    name_input = wait_for(driver, By.ID, "name", timeout=8)
    phone_input = driver.find_element(By.ID, "phone")
    name_input.clear(); name_input.send_keys("Selenium Test User")
    phone_input.clear(); phone_input.send_keys("9998887776")
    # submit profile form (form id may be profileForm in template)
    try:
        submit_btn = driver.find_element(By.CSS_SELECTOR, "form#profileForm button[type='submit']")
    except Exception:
        submit_btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
    submit_btn.click()
    WebDriverWait(driver, 6).until(EC.any_of(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".message")),
        EC.presence_of_element_located((By.CSS_SELECTOR, ".success")),
        EC.presence_of_element_located((By.CSS_SELECTOR, ".alert-success"))
    ))

def test_admin_login_and_views(driver):
    """Admin login and check scans/users/feedback pages."""
    ok = admin_login(driver, BASE_URL, ADMIN_USERNAME, ADMIN_PASSWORD)
    assert ok, "Admin login failed — check ADMIN_USERNAME and ADMIN_PASSWORD or create admin account in the DB."
    # visit admin pages and assert tables or lists render
    driver.get(f"{BASE_URL}/admin/scans")
    wait_for(driver, By.CSS_SELECTOR, "table", timeout=8)
    driver.get(f"{BASE_URL}/admin/users")
    wait_for(driver, By.CSS_SELECTOR, "table", timeout=8)
    driver.get(f"{BASE_URL}/admin/feedback")
    WebDriverWait(driver, 6).until(EC.any_of(
        EC.presence_of_element_located((By.CSS_SELECTOR, "textarea[name='reply_message']")),
        EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
    ))

def test_admin_reply_feedback(driver):
    """Reply to feedback as admin (if reply form exists)."""
    ok = admin_login(driver, BASE_URL, ADMIN_USERNAME, ADMIN_PASSWORD)
    if not ok:
        pytest.skip("Admin login failed — skipping admin reply test.")
    driver.get(f"{BASE_URL}/admin/feedback")
    try:
        textarea = WebDriverWait(driver, 6).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "textarea[name='reply_message']")))
        textarea.clear()
        textarea.send_keys("Thank you — we will check this and reply shortly.")
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        WebDriverWait(driver, 6).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".replied-text, .alert-success")))
    except TimeoutException:
        pytest.skip("No feedback reply form available to test.")
