# ⚽ Football ML Classification

> IS Project | Classification | Python + Flask + TensorFlow

---

## 📁 โครงสร้างโปรเจค

```
football_ml/
├── data/
│   ├── raw/              ← วางไฟล์ Dataset จาก Kaggle ที่นี่
│   └── processed/        ← ไฟล์ที่ผ่าน preprocessing แล้ว
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb   ← Phase 1
│   ├── 02_ensemble_ml.ipynb         ← Phase 2
│   └── 03_neural_network.ipynb      ← Phase 3
├── models/               ← ไฟล์โมเดลที่เทรนแล้ว
├── app/
│   ├── app.py            ← Flask application
│   ├── templates/        ← HTML templates (4 หน้า)
│   └── static/           ← CSS + Images
├── requirements.txt
├── Procfile
└── README.md
```

---

## 🚀 เริ่มต้นใช้งาน

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 2. ดาวน์โหลด Dataset

วางไฟล์ CSV ใน `data/raw/`:

- **FIFA 21**: `fifa21_raw.csv` — [Kaggle Link](https://www.kaggle.com/datasets/yagunnersya/fifa-21-messy-raw-dataset-for-cleaning-exploring)
- **EPL Matches**: `epl_raw.csv` — [Kaggle Link](https://www.kaggle.com/datasets/saife245/english-premier-league)

### 3. รัน Notebooks ตามลำดับ

```bash
jupyter notebook
```

รันตามลำดับ:

1. `01_eda_preprocessing.ipynb` — Preprocessing ทั้ง 2 datasets
2. `02_ensemble_ml.ipynb` — เทรน Ensemble ML (Random Forest + XGBoost + LightGBM + Stacking)
3. `03_neural_network.ipynb` — เทรน Neural Network (MLP)

### 4. รัน Flask App

```bash
python app/app.py
```

เปิด browser ที่ `http://localhost:5000`

---

## ☁️ Deploy บน Render

โปรเจกต์นี้มี `Procfile` พร้อมแล้ว และใช้คำสั่ง start ดังนี้:

```bash
gunicorn --chdir app app:app
```

### 1. เตรียมไฟล์ก่อน push ขึ้น GitHub

- ตรวจให้แน่ใจว่าในโฟลเดอร์ `models/` มีไฟล์เหล่านี้ครบ:
  - `ensemble_model.pkl`
  - `scaler.pkl`
  - `ml_metadata.pkl`
  - `nn_model.keras`
  - `scaler_nn.pkl`
  - `nn_metadata.pkl`
- โปรเจกต์นี้ต้องใช้ไฟล์โมเดลตอน runtime ถ้าไม่ push ขึ้น repo หน้า demo จะตอบกลับ `503`

### 2. Push โปรเจกต์ขึ้น GitHub

```bash
git add .
git commit -m "prepare project for deploy"
git push
```

### 3. สร้าง Web Service บน Render

ไปที่ Render Dashboard แล้วเลือก `New > Web Service` จากนั้นเชื่อม GitHub repo นี้

ค่าที่แนะนำ:

- `Environment`: `Python 3`
- `Python Version`: ใช้จากไฟล์ `.python-version` ซึ่งโปรเจกต์นี้กำหนดเป็น `3.11.11`
- `Build Command`: `pip install -r requirements.txt`
- `Start Command`: `gunicorn --chdir app app:app`

### 4. หลัง deploy เสร็จ

- Render จะสร้าง URL ลักษณะ `https://your-service-name.onrender.com`
- ทดสอบหน้าเหล่านี้:
  - `/`
  - `/ml`
  - `/nn`
  - `/demo/ml`
  - `/demo/nn`

### 5. หมายเหตุสำคัญ

- Render ต้องการให้ web service bind กับ `0.0.0.0` และใช้พอร์ตจาก environment ซึ่งโปรเจกต์นี้รองรับแล้ว
- ถ้าไม่กำหนด Python version, Render service ใหม่อาจใช้ Python `3.14.x` ซึ่งยังไม่รองรับ TensorFlow ของโปรเจกต์นี้
- ถ้า build ช้าในครั้งแรกเป็นเรื่องปกติ เพราะมีแพ็กเกจ ML/DL เช่น TensorFlow, XGBoost และ LightGBM
- ถ้า deploy สำเร็จแต่กดทำนายไม่ได้ ให้เช็ก log ก่อนว่าโหลดไฟล์ใน `models/` ครบหรือไม่

### 6. ถ้า service บน Render ถูกสร้างไปแล้ว

- เข้าไปที่ `Environment` แล้วเพิ่มตัวแปร `PYTHON_VERSION=3.11.11`
- จากนั้นกด `Manual Deploy` หรือ `Clear build cache & deploy` ใหม่อีกครั้ง

---

## 🤖 โมเดล

| โมเดล                  | Dataset     | Task                            | Library                         |
| ---------------------- | ----------- | ------------------------------- | ------------------------------- |
| Ensemble ML (Stacking) | FIFA 21     | Player Position: GK/DEF/MID/FWD | scikit-learn, XGBoost, LightGBM |
| Neural Network (MLP)   | EPL Matches | Match Result: Win/Draw/Loss     | TensorFlow/Keras                |

---

## 🌐 Web Application (4 หน้า)

| หน้า             | URL        | เนื้อหา                          |
| ---------------- | ---------- | -------------------------------- |
| Machine Learning | `/ml`      | ทฤษฎี + Preprocessing + Metrics  |
| Neural Network   | `/nn`      | ทฤษฎี + Architecture + Results   |
| Demo ML          | `/demo/ml` | กรอก FIFA stats → ทำนาย Position |
| Demo NN          | `/demo/nn` | กรอก EPL stats → ทำนายผลแข่ง     |

---

## 👥 ทีม

| คนที่   | ความรับผิดชอบ                                                   |
| ------- | --------------------------------------------------------------- |
| คนที่ 1 | FIFA 21 preprocessing + Ensemble ML + หน้า ML + Demo ML         |
| คนที่ 2 | EPL preprocessing + Neural Network + หน้า NN + Demo NN + Deploy |
