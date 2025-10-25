#  Smart Student Dashboard

https://jyotishman89-smart-student-dashboard-ssd-r8odh3.streamlit.app/

A **personal academic dashboard** built with [Streamlit](https://streamlit.io/) to help me track:

- Marks across 4 exams (Sessional 1, Mid Term, Sessional 2, End Term)
- SGPA & CGPA (auto-calculated with credit weighting)
- Interactive charts (bar, pie, line with smooth animations)
- Attendance tracking with next-class advice (whether you can skip or not)
- Snapshots & history (see past performance over time)

---

##  Motivation
Managing marks, attendance, and SGPA manually was a hassle for me.  
I wanted **one place** where I could:

- Enter my marks & attendance
- Get **dynamic charts** instead of plain tables
- Know instantly if I could **skip the next class safely**
- Track **SGPA/CGPA over semesters**

So I built this dashboard to **solve my own problem** and later made it open-source so others can have experience of it too.

---

*Note: This project was built for personal use and learning purposes. It’s not optimized for large-scale data or advanced security.*

---

##  Features
● Secure login with Roll No. and password  
● Dynamic animated charts (bar, pie, line)  
● Attendance predictor (skip vs attend advice)  
● SGPA & CGPA calculation (with credit weighting)  
● Save and restore snapshots across semesters  
● Dark theme, glassmorphism UI  

---

##  Tech Stack
- **Frontend/Backend**: [Streamlit](https://streamlit.io/)  
- **Data handling**: Pandas, NumPy  
- **Visuals**: Plotly  
- **Persistence**: CSV (can be extended to databases like SQLite/Postgres)

---
##  Clone the repo
```bash
git clone https://github.com/Jyotishman89/Smart-Student-Dashboard.git
cd Smart-Student-Dashboard 
```

##  Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

##  Run the app
```bash
streamlit run ssd.py
```
---

##  Disclaimer
This project has been developed solely for **personal use and learning purposes**.

It is intended to handle **limited, individual-level data** and is **not optimized for large-scale or multi-user environments**. 

Please note that **data security and authentication mechanisms are minimal** in this version, as the focus is primarily on functionality and design rather than robust security implementation.Future updates may include improved scalability and enhanced security features.


---





