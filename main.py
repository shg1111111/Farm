import os
import time
import random
import json
import cv2
import numpy as np
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from urllib.parse import quote
import logging
import traceback
import sqlite3
import requests

# ============== إعدادات التسجيل والمراقبة ==============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ad_farm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============== إعدادات المجلدات والملفات ==============
TEMPLATES_DIR = "ad_templates"
DB_FILE = "ad_farm.db"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# ============== إعداد قاعدة البيانات ==============
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # جدول الجلسات
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        platform TEXT NOT NULL,
        start_time DATETIME NOT NULL,
        end_time DATETIME,
        duration REAL,
        status TEXT NOT NULL,
        earnings REAL DEFAULT 0,
        proxy_used BOOLEAN,
        details TEXT
    )
    ''')
    
    # جدول التعلم
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS learning (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        action TEXT NOT NULL,
        ad_platform TEXT NOT NULL,
        position TEXT,
        detection_method TEXT,
        result TEXT NOT NULL
    )
    ''')
    
    # جدول الإعلانات
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        platform TEXT NOT NULL,
        detected_at DATETIME NOT NULL,
        action_taken TEXT NOT NULL,
        revenue_generated REAL DEFAULT 0
    )
    ''')
    
    conn.commit()
    conn.close()

# ============== فئة الكشف البصري المتقدمة ==============
class AdvancedAdDetector:
    def __init__(self):
        self.templates = self.load_templates()
        self.last_detection = None
        
    def load_templates(self):
        templates = {}
        template_sizes = [(25, 25), (30, 30), (40, 40), (50, 50)]
        
        for file in os.listdir(TEMPLATES_DIR):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = cv2.imread(os.path.join(TEMPLATES_DIR, file), 0)
                    if img is not None:
                        for size in template_sizes:
                            try:
                                resized = cv2.resize(img, size)
                                key = f"{file}_{size[0]}x{size[1]}"
                                templates[key] = resized
                            except:
                                continue
                except Exception as e:
                    logger.error(f"Error loading template {file}: {str(e)}")
        return templates
    
    def detect_ad(self, driver):
        """الكشف المتقدم عن الإعلانات بطرق متعددة"""
        try:
            # 1. الكشف البصري
            screenshot = driver.get_screenshot_as_png()
            screenshot_np = np.frombuffer(screenshot, np.uint8)
            screenshot_cv = cv2.imdecode(screenshot_np, cv2.IMREAD_COLOR)
            
            close_button_pos = self.detect_close_button(screenshot_cv)
            if close_button_pos:
                self.last_detection = ("visual", close_button_pos)
                return close_button_pos, "visual"
            
            # 2. الكشف باستخدام الذكاء الاصطناعي (نموذج بسيط)
            ai_detection = self.ai_based_detection(screenshot_cv)
            if ai_detection:
                self.last_detection = ("ai", ai_detection)
                return ai_detection, "ai_model"
            
            # 3. الكشف باستخدام DOM
            dom_position = self.dom_based_detection(driver)
            if dom_position:
                self.last_detection = ("dom", dom_position)
                return dom_position, "dom"
            
            # 4. الكشف باستخدام الشاشة الكاملة
            fullscreen_position = self.fullscreen_detection(driver)
            if fullscreen_position:
                self.last_detection = ("fullscreen", fullscreen_position)
                return fullscreen_position, "fullscreen"
            
            return None, "not_found"
        
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return None, "error"
    
    def detect_close_button(self, screenshot):
        """الكشف عن زر الإغلاق باستخدام قوالب متعددة"""
        try:
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            for name, template in self.templates.items():
                if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
                    continue
                    
                res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                
                if max_val > 0.78:  # زيادة الحساسية
                    top_left = max_loc
                    center_x = top_left[0] + template.shape[1] // 2
                    center_y = top_left[1] + template.shape[0] // 2
                    return (center_x, center_y)
        except Exception as e:
            logger.error(f"Button detection error: {str(e)}")
        return None
    
    def ai_based_detection(self, screenshot):
        """نموذج بسيط للكشف باستخدام خصائص الصورة"""
        try:
            # تحويل الصورة إلى مساحة لون HSV
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            
            # كشف الألوان الشائعة في أزرار الإغلاق (أحمر، أبيض، أسود)
            lower_red = np.array([0, 120, 70])
            upper_red = np.array([10, 255, 255])
            mask_red = cv2.inRange(hsv, lower_red, upper_red)
            
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 30])
            mask_black = cv2.inRange(hsv, lower_black, upper_black)
            
            # دمج الأقنعة
            combined_mask = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_white, mask_black))
            
            # البحث عن كونتورات
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # تصفية حسب الحجم والموقع
                if 20 < w < 100 and 20 < h < 100:
                    if (screenshot.shape[1] - (x + w) < 50) and (y < 100):
                        return (x + w // 2, y + h // 2)
        
        except Exception as e:
            logger.error(f"AI detection error: {str(e)}")
        return None
    
    def dom_based_detection(self, driver):
        """الكشف عن الإعلانات باستخدام عناصر DOM"""
        selectors = [
            'div[class*="close"][class*="btn"], [class*="close"][class*="button"]',
            'div[id*="close"][id*="btn"], [id*="close"][id*="button"]',
            'div[class*="dismiss"][class*="btn"], [class*="dismiss"][class*="button"]',
            'div[class*="skip"][class*="btn"], [class*="skip"][class*="button"]',
            'div[class*="exit"][class*="btn"], [class*="exit"][class*="button"]',
            'button[aria-label*="close"], button[aria-label*="dismiss"]'
        ]
        
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if element.is_displayed():
                        location = element.location
                        size = element.size
                        return (location['x'] + size['width']//2, location['y'] + size['height']//2)
            except:
                continue
        return None
    
    def fullscreen_detection(self, driver):
        """الكشف عن الإعلانات التي تغطي الشاشة كاملة"""
        try:
            body = driver.find_element(By.TAG_NAME, 'body')
            body_rect = body.rect
            
            for element in driver.find_elements(By.XPATH, "//body/*"):
                try:
                    rect = element.rect
                    if (rect['width'] >= body_rect['width'] * 0.8 and 
                        rect['height'] >= body_rect['height'] * 0.8):
                        # افتراض أن زر الإغلاق في الزاوية العلوية اليمنى
                        return (body_rect['width'] - 30, 30)
                except:
                    continue
        except:
            pass
        return None

# ============== نظام منع الإعلانات المتعددة ==============
def prevent_ads(driver):
    """حقن كود JavaScript لمنع ظهور إعلانات متعددة"""
    try:
        js_code = """
        // منع الإعلانات المنبثقة المتعددة
        let adShown = false;
        
        const observer = new MutationObserver(mutations => {
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === 1) {
                        const styles = window.getComputedStyle(node);
                        const zIndex = parseInt(styles.zIndex) || 0;
                        
                        // اكتشاف عناصر الإعلانات
                        if ((node.id && (node.id.includes('ad') || node.id.includes('popup'))) ||
                            (node.className && (node.className.includes('ad') || node.className.includes('popup'))) ||
                            (zIndex > 1000 && node.offsetWidth > window.innerWidth * 0.7)) {
                            
                            if (adShown) {
                                node.remove();
                                console.log('تم إزالة إعلان متعدد');
                            } else {
                                adShown = true;
                                console.log('تم اكتشاف إعلان أولي');
                            }
                        }
                    }
                });
            });
        });
        
        observer.observe(document.body, { childList: true, subtree: true });
        """
        driver.execute_script(js_code)
        return True
    except Exception as e:
        logger.error(f"Failed to inject ad prevention: {str(e)}")
        return False

# ============== نظام التعلم الآلي ==============
class AdDecisionSystem:
    def __init__(self):
        self.conn = sqlite3.connect(DB_FILE)
    
    def get_optimal_action(self, platform, detection_method):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT action, COUNT(*) as count,
                       SUM(CASE WHEN result = 'success' THEN 1 ELSE 0 END) as success_count
                FROM learning
                WHERE ad_platform = ? AND detection_method = ?
                GROUP BY action
            ''', (platform, detection_method))
            
            actions = {}
            total_success = 0
            total_attempts = 0
            
            for row in cursor.fetchall():
                action, attempts, successes = row
                success_rate = successes / attempts if attempts > 0 else 0
                actions[action] = success_rate
                total_success += successes
                total_attempts += attempts
            
            # إذا لم توجد بيانات كافية
            if not actions:
                return "click" if random.random() > 0.6 else "close"
            
            # اختيار الإجراء الأفضل
            best_action = max(actions, key=actions.get)
            
            # إذا كان الفرق بسيط، اختيار عشوائي
            if total_attempts > 20 and max(actions.values()) - min(actions.values()) < 0.2:
                return "click" if random.random() > 0.5 else "close"
            
            return best_action
        
        except Exception as e:
            logger.error(f"Decision system error: {str(e)}")
            return "click" if random.random() > 0.5 else "close"
    
    def record_result(self, session_id, timestamp, action, platform, position, method, result):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO learning (session_id, timestamp, action, ad_platform, position, detection_method, result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, timestamp, action, platform, json.dumps(position), method, result))
            self.conn.commit()
            
            # تسجيل الإيرادات (تقديري)
            if result == "success" and action == "click":
                revenue = round(random.uniform(0.001, 0.015), 5)  # إيراد بين 0.001 و 0.015 دولار
                cursor.execute('''
                    INSERT INTO ads (platform, detected_at, action_taken, revenue_generated)
                    VALUES (?, ?, ?, ?)
                ''', (platform, timestamp, action, revenue))
                self.conn.commit()
                return revenue
            return 0
        except Exception as e:
            logger.error(f"Failed to record learning: {str(e)}")
            return 0

# ============== نظام التفاعل البشري ==============
class HumanLikeInteraction:
    @staticmethod
    def simulate_interaction(driver):
        try:
            body = driver.find_element(By.TAG_NAME, 'body')
            actions = ActionChains(driver)
            
            # حركات الماوس العشوائية
            for _ in range(random.randint(3, 6)):
                x = random.randint(50, driver.get_window_size()['width'] - 50)
                y = random.randint(50, driver.get_window_size()['height'] - 50)
                actions.move_to_element_with_offset(body, x, y)
                actions.pause(random.uniform(0.1, 0.3))
            
            # التمرير
            scroll_amount = random.randint(300, 800)
            driver.execute_script(f"window.scrollBy(0, {scroll_amount})")
            actions.pause(random.uniform(0.5, 1.5))
            
            # نقرات عشوائية
            if random.random() > 0.3:
                elements = driver.find_elements(By.CSS_SELECTOR, "a, button, div[onclick]")
                if elements:
                    element = random.choice(elements)
                    try:
                        if element.is_displayed() and element.is_enabled():
                            actions.move_to_element(element)
                            actions.pause(0.2)
                            actions.click()
                            actions.perform()
                            time.sleep(random.uniform(1, 3))
                    except:
                        pass
            
            actions.perform()
            return True
        except Exception as e:
            logger.error(f"Human interaction failed: {str(e)}")
            return False

# ============== نظام إدارة المتصفح ==============
class StealthBrowser:
    @staticmethod
    def create_driver(proxy_url=None):
        chrome_options = Options()
        
        # إعدادات التخفي
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-webgl")
        chrome_options.add_argument("--disable-canvas")
        
        # وكيل مستخدم عشوائي
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/116.0",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1"
        ]
        chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
        
        # إعدادات النافذة
        width = random.choice([1366, 1440, 1536, 1920])
        height = random.choice([768, 900, 1080])
        chrome_options.add_argument(f"--window-size={width},{height}")
        
        # وضع headless للتشغيل على السيرفر
        chrome_options.add_argument("--headless=new")
        
        # إعداد البروكسي
        if proxy_url:
            chrome_options.add_argument(f'--proxy-server={proxy_url}')
        
        # إنشاء المتصفح
        driver = webdriver.Chrome(options=chrome_options)
        
        # تعطيل خصائص التشغيل الآلي
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.execute_script("""
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            window.chrome = undefined;
        """)
        
        return driver

# ============== نظام إدارة الجلسات ==============
class SessionManager:
    def __init__(self, website_url, platforms, scraperapi_key=None):
        self.website_url = website_url
        self.platforms = platforms
        self.scraperapi_key = scraperapi_key
        self.decision_system = AdDecisionSystem()
        self.total_earnings = 0
        self.session_count = 0
        self.conn = sqlite3.connect(DB_FILE)
    
    def run_session(self, platform):
        session_id = f"{platform}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        start_time = datetime.now()
        status = "failed"
        earnings = 0
        details = ""
        proxy_used = bool(self.scraperapi_key)
        
        try:
            # تحضير الرابط
            if self.scraperapi_key:
                proxy_url = f"http://scraperapi.retry_503=true:{self.scraperapi_key}@proxy-server.scraperapi.com:8001"
                target_url = f"http://api.scraperapi.com/?api_key={self.scraperapi_key}&url={quote(self.website_url)}&render=true"
            else:
                proxy_url = None
                target_url = self.website_url
            
            # إنشاء المتصفح
            driver = StealthBrowser.create_driver(proxy_url)
            detector = AdvancedAdDetector()
            
            # تسجيل بدء الجلسة
            logger.info(f"🚀 Starting session {session_id} for {platform}")
            
            # زيارة الموقع
            driver.get(target_url)
            logger.info(f"🌐 Visited: {self.website_url}")
            
            # تفاعل بشري أولي
            HumanLikeInteraction.simulate_interaction(driver)
            time.sleep(random.uniform(5, 10))
            
            # منع الإعلانات المتعددة
            prevent_ads(driver)
            
            # الانتظار لظهور الإعلان
            ad_wait = random.uniform(20, 30)
            logger.info(f"⏳ Waiting {ad_wait:.1f}s for ads...")
            time.sleep(ad_wait)
            
            # الكشف عن الإعلان
            ad_position, detection_method = detector.detect_ad(driver)
            
            if ad_position:
                logger.info(f"🪧 Ad detected ({detection_method}) at {ad_position}")
                
                # تحديد الإجراء الأمثل
                action = self.decision_system.get_optimal_action(platform, detection_method)
                
                # تنفيذ الإجراء
                body = driver.find_element(By.TAG_NAME, 'body')
                actions = ActionChains(driver)
                actions.move_to_element_with_offset(body, ad_position[0], ad_position[1])
                actions.pause(random.uniform(0.3, 1.0))
                
                if action == "click":
                    actions.click()
                    logger.info("💸 Clicked on ad (revenue generating)")
                    
                    # انتظار تحميل صفحة الإعلان
                    time.sleep(random.uniform(8, 15))
                    
                    # العودة إلى الموقع الأصلي
                    try:
                        driver.back()
                        time.sleep(random.uniform(3, 7))
                    except WebDriverException:
                        driver.get(self.website_url)
                    
                    # تسجيل النجاح
                    result = "success"
                else:
                    actions.click()
                    logger.info("✅ Clicked close button")
                    result = "closed"
                
                actions.perform()
                
                # تسجيل النتيجة
                earnings = self.decision_system.record_result(
                    session_id, 
                    datetime.now().isoformat(), 
                    action, 
                    platform, 
                    ad_position, 
                    detection_method, 
                    result
                )
                
                if result == "success":
                    self.total_earnings += earnings
                    status = "success"
                else:
                    status = "ad_closed"
            else:
                logger.info("🔍 No ad detected")
                status = "no_ad"
            
            # تفاعل بعد الإعلان
            if status == "success":
                HumanLikeInteraction.simulate_interaction(driver)
                stay_time = random.uniform(25, 45)
                logger.info(f"🕒 Staying for {stay_time:.1f}s")
                time.sleep(stay_time)
            else:
                stay_time = random.uniform(10, 20)
                logger.info(f"🕒 Short stay for {stay_time:.1f}s")
                time.sleep(stay_time)
            
            # تسجيل نجاح الجلسة
            logger.info(f"✅ Session {session_id} completed")
            status = "completed"
        
        except Exception as e:
            logger.error(f"❌ Session {session_id} failed: {str(e)}")
            details = traceback.format_exc()
            status = "error"
        
        finally:
            try:
                if driver:
                    driver.quit()
            except:
                pass
            
            # تسجيل الجلسة في قاعدة البيانات
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            try:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO sessions (session_id, platform, start_time, end_time, duration, status, earnings, proxy_used, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, platform, start_time, end_time, duration, status, earnings, proxy_used, details))
                self.conn.commit()
            except Exception as e:
                logger.error(f"Failed to save session: {str(e)}")
            
            self.session_count += 1
            return status

    def start_farm(self, sessions_per_platform):
        total_sessions = len(self.platforms) * sessions_per_platform
        completed = 0
        start_time = datetime.now()
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🚀 Starting ad farm for {self.website_url}")
        logger.info(f"📢 Platforms: {', '.join(self.platforms)}")
        logger.info(f"🔁 Sessions per platform: {sessions_per_platform}")
        logger.info(f"🔁 Total sessions: {total_sessions}")
        logger.info(f"{'='*50}\n")
        
        # دورة الجلسات
        while completed < total_sessions:
            platform = random.choice(self.platforms)
            session_status = self.run_session(platform)
            
            if session_status != "error":
                completed += 1
            
            # حساب وقت الانتظار الديناميكي
            if completed < total_sessions:
                base_delay = max(180, min(600, 300 * (completed / total_sessions)))
                jitter = random.uniform(-0.2, 0.2) * base_delay
                delay = max(60, base_delay + jitter)
                
                logger.info(f"⏳ Next session in {delay:.1f}s")
                time.sleep(delay)
        
        # تقرير نهائي
        elapsed = datetime.now() - start_time
        avg_earning = self.total_earnings / total_sessions if total_sessions > 0 else 0
        total_earning = self.total_earnings
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🏁 Farm completed! Total sessions: {total_sessions}")
        logger.info(f"⏱️ Total time: {str(elapsed)}")
        logger.info(f"💰 Estimated revenue: ${total_earning:.5f}")
        logger.info(f"💵 Average per session: ${avg_earning:.5f}")
        logger.info(f"{'='*50}")
        
        # إرسال تقرير بالبريد (اختياري)
        self.send_report(total_sessions, elapsed, total_earning)
        
        return total_earning
    
    def send_report(self, total_sessions, elapsed, total_earning):
        """إرسال تقرير بالنتائج (يمكن تفعيله لاحقا)"""
        # يمكن إضافة كود إرسال تقرير هنا
        pass

# ============== التكوين الرئيسي ==============
if __name__ == "__main__":
    # تهيئة قاعدة البيانات
    init_db()
    
    # ============== إعدادات المزرعة ==============
    # 1. رابط موقعك
    WEBSITE_URL = "https://tpmscool.web.app"  # استبدل برابط موقعك
    
    # 2. منصات الإعلانات
    AD_PLATFORMS = ["propellerads", "popads", "monetag"]
    
    # 3. مفتاح ScraperAPI (اختياري)
    SCRAPERAPI_KEY = "your_scraperapi_key_here"  # استبدل بمفتاحك
    
    # 4. عدد الجلسات لكل منصة
    SESSIONS_PER_PLATFORM = 60
    
    # ============== تشغيل المزرعة ==============
    manager = SessionManager(
        website_url=WEBSITE_URL,
        platforms=AD_PLATFORMS,
        scraperapi_key=SCRAPERAPI_KEY if SCRAPERAPI_KEY != "e82c11cd98307d69928f0a5fdd713d6b" else None
    )
    
    manager.start_farm(SESSIONS_PER_PLATFORM)
