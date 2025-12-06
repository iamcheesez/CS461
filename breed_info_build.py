import json
import re

# โหลดไฟล์ (สมมติว่าคุณมีไฟล์ต้นฉบับชื่อ breed_info_en.json)
# ถ้าไม่มีไฟล์จริง ให้ข้ามส่วนโหลดไฟล์และดูที่ตัวแปร json_data ด้านล่าง
input_filename = 'breed_info_en.json'
output_filename = 'breed_info_th_advanced.json'

# --- ส่วนของการตั้งค่าการแปล ---

# 1. การจัดเรียงประโยค (สำคัญที่สุด: สลับตำแหน่งคำขยาย)
# Pattern: "is a [คำขยาย] breed of dog" -> "เป็นสายพันธุ์สุนัขที่ [คำขยาย]"
structural_patterns = [
    (r"is a (.+?) breed of dog", r"เป็นสายพันธุ์สุนัขที่มีลักษณะ\1"),
    (r"is a (.+?) breed", r"เป็นสายพันธุ์ที่\1"),
    (r"The (.+?) is a", r"\1 เป็น"),
    (r"that was originally bred in (.+?) for (.+?)", r"ซึ่งเดิมทีเพาะพันธุ์ใน\1 เพื่อ\2"),
    (r"They are known for (.+?)", r"พวกมันเป็นที่รู้จักในเรื่อง\1"),
    (r"They are (.+?)", r"พวกมันมีนิสัย\1"),
]

# 2. คลังคำศัพท์ (เรียงจากกลุ่มคำยาว -> คำสั้น)
vocabulary = {
    # คำขยายรูปร่าง
    "large and muscular": "ขนาดใหญ่และมีกล้ามเนื้อ",
    "large and powerful": "ขนาดใหญ่และทรงพลัง",
    "small and playful": "ขนาดเล็กและขี้เล่น",
    "small to medium-sized": "ขนาดเล็กถึงปานกลาง",
    "medium-sized": "ขนาดปานกลาง",
    "large": "ขนาดใหญ่",
    "small": "ขนาดเล็ก",
    "muscular": "มีกล้ามเนื้อ",
    "athletic": "หุ่นนักกีฬา/ปราดเปรียว",
    "slender": "ผอมเพรียว",
    "elegant": "สง่างาม",
    "sturdy": "บึกบึน",
    
    # ประเทศ
    "Canada": "แคนาดา", "Germany": "เยอรมนี", "England": "อังกฤษ", 
    "Japan": "ญี่ปุ่น", "United States": "สหรัฐอเมริกา", "France": "ฝรั่งเศส",
    "Scotland": "สกอตแลนด์", "Ireland": "ไอร์แลนด์", "Wales": "เวลส์",
    "China": "จีน", "Russia": "รัสเซีย", "Italy": "อิตาลี",
    "Spain": "สเปน", "Portugal": "โปรตุเกส", "Netherlands": "เนเธอร์แลนด์",
    "Belgium": "เบลเยียม", "Australia": "ออสเตรเลีย", "Africa": "แอฟริกา",
    
    # กิจกรรม/หน้าที่
    "retrieving game": "การคาบสัตว์ที่ล่าได้กลับมา",
    "hunting small game": "การล่าสัตว์เล็ก",
    "hunting large game": "การล่าสัตว์ใหญ่",
    "hunting waterfowl": "การล่านกเป็ดน้ำ",
    "hunting": "การล่าสัตว์",
    "herding and guarding livestock": "การต้อนและเฝ้าปศุสัตว์",
    "guarding and protecting property": "การเฝ้าและปกป้องทรัพย์สิน",
    "working on farms": "การใช้งานในฟาร์ม",
    
    # นิสัย
    "intelligent": "ฉลาด",
    "friendly": "เป็นมิตร",
    "eager to please": "ชอบเอาใจเจ้าของ",
    "loyal": "ซื่อสัตย์",
    "gentle": "อ่อนโยน",
    "protective": "หวงแหน/ปกป้อง",
    "energetic": "กระตือรือร้น",
    "affectionate": "ขี้อ้อน/น่ารักใคร่",
    "playful": "ขี้เล่น",
    "independent": "รักอิสระ",
    "alert": "ตื่นตัว",
    
    # คำทั่วไป
    "make excellent family pets": "เป็นสัตว์เลี้ยงครอบครัวที่ยอดเยี่ยม",
    "make excellent companion dogs": "เป็นสุนัขคู่ใจที่ยอดเยี่ยม",
    "make excellent guard dogs": "เป็นสุนัขเฝ้ายามชั้นเลิศ",
    "double coat": "ขนสองชั้น",
    "smooth coat": "ขนเรียบ",
    "wiry coat": "ขนหยาบ",
    "long coat": "ขนยาว",
    
    # คำเชื่อม/อื่นๆ
    " and ": " และ",
    ", and ": " และ",
    ". ": ". ",
    "The ": "", # ลบ The นำหน้าทิ้ง
}

def smart_translate(text, breed_name):
    if not text: return ""
    
    # 1. ลบชื่อพันธุ์ออกจากประโยคก่อน (เพื่อไม่ให้แปลชื่อพันธุ์) แล้วใส่ Placeholder ไว้
    # แต่เนื่องจากเราต้องการคงชื่ออังกฤษไว้ วิธีง่ายที่สุดคือ ลบ "The " นำหน้าชื่อออก
    text = text.replace("The " + breed_name, breed_name)
    
    # 2. ใช้ Regex ปรับโครงสร้างประโยค (Structure Mapping)
    for pattern, replacement in structural_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # 3. แทนที่คำศัพท์ (Vocabulary Mapping)
    for eng, thai in vocabulary.items():
        # ใช้ re.sub เพื่อให้ match ทั้งคำ (ป้องกันการแทนที่คำที่ซ้อนกัน)
        # หรือใช้ replace ธรรมดาถ้ามั่นใจลำดับ dictionary
        text = text.replace(eng, thai)
        text = text.replace(eng.lower(), thai) # รองรับตัวพิมพ์เล็ก
        
    # 4. เก็บตกคำที่ยังหลงเหลือ (Cleanup)
    text = text.replace("are ", "") # ลบ are ที่อาจหลงเหลือ
    text = text.replace("  ", " ") # ลบช่องว่างซ้ำ
    
    # ปรับรูปประโยคภาษาไทยให้ลื่นไหลขึ้น
    text = text.replace("เป็นสายพันธุ์สุนัขที่มีลักษณะขนาด", "เป็นสุนัขพันธุ์") 
    text = text.replace("เป็นสายพันธุ์สุนัขที่มีลักษณะ", "เป็นสุนัขพันธุ์")
    
    return text

# --- ส่วนของการรันโปรแกรม ---

try:
    with open(input_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for item in data['data']:
        original_desc = item['attributes']['description']
        breed_name = item['attributes']['name']
        
        # เรียกใช้ฟังก์ชันแปล
        translated = smart_translate(original_desc, breed_name)
        item['attributes']['description'] = translated

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ แปลเสร็จเรียบร้อย! บันทึกไฟล์ที่: {output_filename}")
    
    # Test print เพื่อดูผลลัพธ์ของ Labrador (ถ้ามีในไฟล์)
    # หรือลองพิมพ์ตัวอย่างจาก Input ของคุณ
    sample_text = "The Labrador Retriever is a large and muscular breed of dog that was originally bred in Canada for retrieving game. They are intelligent, friendly, and eager to please, and make excellent family pets."
    print("\n--- ตัวอย่างผลลัพธ์ (Labrador) ---")
    print(smart_translate(sample_text, "Labrador Retriever"))

except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์ {input_filename} กรุณาตรวจสอบชื่อไฟล์ต้นฉบับ")