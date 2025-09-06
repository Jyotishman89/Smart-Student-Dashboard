# 🎓 Smart Student Dashboard

A **personal academic dashboard** built with [Streamlit](https://streamlit.io/) to help me track:

- 📚 Marks across 4 exams (Sessional 1, Mid Term, Sessional 2, End Term)
- 🧮 SGPA & CGPA (auto-calculated with credit weighting)
- 📈 Interactive charts (bar, pie, line with smooth animations)
- 🕒 Attendance tracking with next-class advice (whether you can skip or not)
- 💾 Snapshots & history (see past performance over time)

---

## 🔹 Motivation
Managing marks, attendance, and SGPA manually was a hassle for me.  
I wanted **one place** where I could:

- Enter my marks & attendance
- Get **dynamic charts** instead of plain tables
- Know instantly if I could **skip the next class safely**
- Track **SGPA/CGPA over semesters**

So I built this dashboard to **solve my own problem** and later made it open-source so others can use it too.

---

## 🔹 Features
✅ Secure login with Roll No. and password  
✅ Dynamic animated charts (bar, pie, line)  
✅ Attendance predictor (skip vs attend advice)  
✅ SGPA & CGPA calculation (with credit weighting)  
✅ Save and restore snapshots across semesters  
✅ Dark theme, glassmorphism UI  

---

## 🔹 Tech Stack
- **Frontend/Backend**: [Streamlit](https://streamlit.io/)  
- **Data handling**: Pandas, NumPy  
- **Visuals**: Plotly  
- **Persistence**: CSV (can be extended to databases like SQLite/Postgres)

---
## 1️⃣ Clone the repo
```bash
git clone https://github.com/Jyotishman89/Smart-Student-Dashboard.git
cd Smart-Student-Dashboard 
```

## 2️⃣ Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3️⃣ Run the app
```bash
streamlit run ssd.py
```






