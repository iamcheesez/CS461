# 🐕 What DAWG!? - Dog Breed Classifier

<p align="center">
  <img src="static/logo.png" alt="What DAWG!? Logo" width="200">
</p>

<p align="center">
  <strong>ระบบจำแนกสายพันธุ์สุนัขด้วย Deep Learning</strong><br>
  รองรับ 120 สายพันธุ์ | ภาษาไทย/อังกฤษ | ข้อมูลสายพันธุ์ครบถ้วน
</p>

---

## 📋 สารบัญ

- [Features](#-features)
- [Demo](#-demo)
- [การติดตั้ง](#-การติดตั้ง)
- [การใช้งาน](#-การใช้งาน)
- [โครงสร้างโปรเจค](#-โครงสร้างโปรเจค)
- [Tech Stack](#-tech-stack)
- [Model Performance](#-model-performance)

---

## ✨ Features

| Feature | รายละเอียด |
|---------|-----------|
| 🎯 **จำแนกสายพันธุ์** | รองรับ 120 สายพันธุ์สุนัข ด้วย ResNet50 |
| 📊 **Top-5 Predictions** | แสดงผลลัพธ์ 5 อันดับแรกพร้อม % ความมั่นใจ |
| 🌐 **2 ภาษา** | รองรับภาษาไทยและอังกฤษ สลับได้ทันที |
| 📖 **ข้อมูลสายพันธุ์** | นิสัย, ขนาด, น้ำหนัก, อายุขัย (283 สายพันธุ์) |
| 💬 **ระบบ Feedback** | ผู้ใช้สามารถแก้ไขผลลัพธ์เพื่อปรับปรุงโมเดล |
| 📱 **Responsive** | ใช้งานได้ทุกอุปกรณ์ |

---

## 🎬 Demo

<p align="center">
  <img src="docs/demo.gif" alt="Demo" width="600">
</p>

**ลองใช้งาน:** อัปโหลดรูปสุนัข → รับผลการจำแนกสายพันธุ์ทันที!

---

## 🚀 การติดตั้ง

### ความต้องการของระบบ

- Python 3.9+
- RAM อย่างน้อย 4GB
- พื้นที่ดิสก์ ~500MB (รวมโมเดล)

### ขั้นตอนการติดตั้ง

#### 1. Clone Repository

```bash
git clone https://github.com/iamcheesez/CS461.git
cd CS461
```

#### 2. สร้าง Virtual Environment (แนะนำ)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

#### 4. ตรวจสอบไฟล์ที่จำเป็น

ตรวจสอบว่ามีไฟล์เหล่านี้:

```
models/
├── best_model.pth        # โมเดลที่ train แล้ว (~100MB)
└── class_mapping.json    # Mapping ของ class

breed_info_en.json        # ข้อมูลสายพันธุ์ (EN)
breed_info_th.json        # ข้อมูลสายพันธุ์ (TH)

static/
└── logo.png              # โลโก้เว็บ
```

---

## 💻 การใช้งาน

### วิธีที่ 1: รันเว็บแอป (แนะนำ)

```bash
python app.py
```

จากนั้นเปิด Browser ไปที่: **http://localhost:5000**

### วิธีที่ 2: ใช้ Command Line

```bash
# ทำนายรูปเดียว
python predict.py path/to/dog_image.jpg

# ทำนายหลายรูป
python predict.py image1.jpg image2.jpg image3.jpg
```

### วิธีที่ 3: ใช้ใน Python Script

```python
from app import DogBreedPredictor

# สร้าง predictor
predictor = DogBreedPredictor()

# ทำนาย
results = predictor.predict("path/to/dog.jpg", top_k=5)

for breed, confidence in results:
    print(f"{breed}: {confidence:.2%}")
```

---

## 📁 โครงสร้างโปรเจค

```
CS461/
├── 📜 app.py                  # Flask Web Application
├── 📜 wsgi.py                 # WSGI entry point (production)
├── 📜 train.py                # Script สำหรับ train โมเดล
├── 📜 predict.py              # CLI prediction
├── 📜 dataset.py              # Dataset loader
├── 📜 config.py               # Configuration
├── 📜 retrain.py              # Retrain ด้วย feedback data
│
├── 📂 models/
│   ├── best_model.pth         # Trained model weights
│   ├── class_mapping.json     # Class index mapping
│   └── training_history.json  # Training metrics
│
├── 📂 templates/
│   └── index.html             # Frontend (HTML/CSS/JS)
│
├── 📂 static/
│   └── logo.png               # Logo image
│
├── 📂 uploads/                # Uploaded images (temp)
├── 📂 feedback_data/          # User feedback data
│
├── 📜 breed_info_en.json      # Breed info (English)
├── 📜 breed_info_th.json      # Breed info (Thai)
│
├── 📜 requirements.txt        # Dependencies
├── 📜 requirements-prod.txt   # Production dependencies
├── 📜 Dockerfile              # Docker build
├── 📜 docker-compose.yml      # Docker Compose
├── 📜 Procfile                # Heroku/Railway
└── 📜 README.md               # This file
```

---

## 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| **Backend** | Python, Flask |
| **Deep Learning** | PyTorch, ResNet50 |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Dataset** | Stanford Dogs (120 breeds) |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Architecture** | ResNet50 (Transfer Learning) |
| **Training Epochs** | 20 |
| **Training Accuracy** | 77.06% |
| **Validation Accuracy** | 74.03% |
| **Top-5 Accuracy** | ~95% |
| **Number of Classes** | 120 breeds |

---

## 🔧 การ Train โมเดลใหม่

หากต้องการ train โมเดลใหม่:

```bash
# ต้องมีรูปภาพใน images/Images/
python train.py
```

### การตั้งค่า Training (config.py)

```python
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
```

---

## 🌐 การ Deploy

ดูคู่มือละเอียดที่ [DEPLOYMENT.md](DEPLOYMENT.md)

### Quick Deploy with Docker

```bash
docker-compose up -d --build
```

### Deploy to Railway (ฟรี)

1. Push code to GitHub
2. ไปที่ [railway.app](https://railway.app)
3. New Project → Deploy from GitHub
4. เลือก repo นี้ → Deploy!

---

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | หน้าเว็บหลัก |
| `/predict` | POST | ทำนายสายพันธุ์ (upload image) |
| `/breeds` | GET | รายชื่อสายพันธุ์ทั้งหมด |
| `/breed-info/<name>` | GET | ข้อมูลสายพันธุ์ |
| `/feedback` | POST | ส่ง feedback |
| `/language` | GET/POST | ตั้งค่าภาษา |
| `/health` | GET | Health check |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is for educational purposes (CS461 - Machine Learning).

---

## 👨‍💻 Author

**iamcheesez** - [GitHub](https://github.com/iamcheesez)

---

<p align="center">
  Made with ❤️ and 🐕
</p>
