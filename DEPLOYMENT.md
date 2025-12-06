# 🚀 Deployment Guide - What DAWG!?

## 📋 สิ่งที่ต้องเตรียม

ก่อน deploy ต้องมีไฟล์เหล่านี้:
- `models/best_model.pth` - โมเดลที่ train แล้ว
- `models/class_mapping.json` - Mapping ของ class
- `breed_info_en.json` - ข้อมูลสายพันธุ์ภาษาอังกฤษ
- `breed_info_th.json` - ข้อมูลสายพันธุ์ภาษาไทย
- `static/logo.png` - โลโก้เว็บไซต์

---

## 🐳 Option 1: Docker (แนะนำ)

### Build และ Run ด้วย Docker Compose
```bash
# Build และ start
docker-compose up -d --build

# ดู logs
docker-compose logs -f

# Stop
docker-compose down
```

### Build และ Run ด้วย Docker
```bash
# Build image
docker build -t what-dawg .

# Run container
docker run -d -p 5000:5000 --name what-dawg-app what-dawg

# ดู logs
docker logs -f what-dawg-app
```

เข้าเว็บที่: http://localhost:5000

---

## ☁️ Option 2: Railway.app (ฟรี / ง่ายที่สุด)

1. **สร้าง Account** ที่ https://railway.app
2. **Connect GitHub Repo**
   - New Project → Deploy from GitHub repo
   - เลือก Repository นี้
3. **Deploy อัตโนมัติ**
   - Railway จะ detect `railway.json` และ deploy ให้อัตโนมัติ
4. **ตั้งค่า Domain**
   - Settings → Domains → Generate Domain

**⚠️ ข้อจำกัด:** ไฟล์โมเดล (~100MB) อาจใหญ่เกินไป ต้องใช้ Git LFS

---

## ☁️ Option 3: Render.com (ฟรี)

1. **สร้าง Account** ที่ https://render.com
2. **New Web Service**
   - Connect GitHub repo
   - Build Command: `pip install -r requirements-prod.txt`
   - Start Command: `gunicorn wsgi:app --workers 2 --timeout 120`
3. **ตั้งค่า Disk** (สำหรับเก็บโมเดล)
   - Add Disk → Mount Path: `/app/models`
   - Upload `best_model.pth` และ `class_mapping.json`

---

## ☁️ Option 4: Heroku

1. **ติดตั้ง Heroku CLI**
```bash
# Login
heroku login

# สร้าง app
heroku create what-dawg-app

# Deploy
git push heroku master

# เปิดเว็บ
heroku open
```

**⚠️ ข้อจำกัด:** 
- Free tier ถูกยกเลิกแล้ว (ต้องจ่ายเงิน)
- Slug size limit 500MB

---

## ☁️ Option 5: Google Cloud Run

```bash
# ติดตั้ง gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Login และตั้งค่า project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build และ push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/what-dawg

# Deploy
gcloud run deploy what-dawg \
  --image gcr.io/YOUR_PROJECT_ID/what-dawg \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 120
```

---

## ☁️ Option 6: AWS (EC2 + Docker)

```bash
# SSH เข้า EC2 instance
ssh -i your-key.pem ec2-user@your-ec2-ip

# ติดตั้ง Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Clone repo และ run
git clone https://github.com/iamcheesez/CS461.git
cd CS461
docker-compose up -d --build
```

---

## 🖥️ Option 7: VPS (DigitalOcean, Linode, Vultr)

```bash
# SSH เข้า VPS
ssh root@your-vps-ip

# ติดตั้ง dependencies
apt update && apt upgrade -y
apt install python3 python3-pip python3-venv nginx -y

# Clone repo
git clone https://github.com/iamcheesez/CS461.git
cd CS461

# สร้าง virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-prod.txt

# Run with gunicorn
gunicorn wsgi:app --workers 2 --timeout 120 --bind 0.0.0.0:5000 --daemon

# ตั้งค่า Nginx (optional)
# ดู nginx.conf.example
```

---

## 📁 โครงสร้างไฟล์สำหรับ Deploy

```
CS461/
├── app.py                 # Main Flask app
├── wsgi.py                # WSGI entry point
├── Procfile               # Heroku/Railway
├── runtime.txt            # Python version
├── requirements.txt       # Dev dependencies
├── requirements-prod.txt  # Production dependencies
├── Dockerfile             # Docker build
├── docker-compose.yml     # Docker Compose
├── .dockerignore          # Docker ignore
├── render.yaml            # Render.com config
├── railway.json           # Railway config
├── vercel.json            # Vercel config
├── models/
│   ├── best_model.pth     # Trained model (~100MB)
│   └── class_mapping.json # Class mapping
├── templates/
│   └── index.html         # Frontend
├── static/
│   └── logo.png           # Logo image
├── breed_info_en.json     # EN breed data
└── breed_info_th.json     # TH breed data
```

---

## ⚠️ ข้อควรระวัง

### 1. ขนาดโมเดล
- `best_model.pth` มีขนาด ~100MB
- บาง platform อาจมี limit
- ใช้ Git LFS สำหรับไฟล์ใหญ่

```bash
# ติดตั้ง Git LFS
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add models/best_model.pth
git commit -m "Add model with LFS"
```

### 2. Memory
- โมเดล ResNet50 ต้องใช้ RAM ~1-2GB
- ตั้งค่า memory limit ให้เพียงพอ

### 3. Timeout
- การ load โมเดลครั้งแรกอาจใช้เวลา 30-60 วินาที
- ตั้ง timeout ให้มากพอ (120 วินาที)

### 4. Cold Start
- Serverless platforms มี cold start
- Request แรกอาจช้า

---

## 🔒 Security Checklist

- [ ] ปิด Debug mode (`debug=False`)
- [ ] ใช้ HTTPS
- [ ] ตั้ง SECRET_KEY
- [ ] จำกัดขนาดไฟล์อัปโหลด
- [ ] Validate file types

---

## 📞 Support

หากมีปัญหาในการ deploy สามารถเปิด Issue ใน GitHub repo ได้เลยครับ!
