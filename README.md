# Credit Risk Scoring System

## Overview  
The **Credit Risk Scoring System** is a machine learning project that predicts the likelihood of a loan applicant defaulting on credit.  
It helps financial institutions make **data-driven lending decisions** by assessing customer risk profiles based on historical data.

🔗 **Live Demo:** [https://credit-risk-system.onrender.com](https://credit-risk-system.onrender.com)

---


## Key Features
- 🧩 **Machine Learning Model** — Logistic Regression-based classifier for credit risk prediction  
- ⚙️ **FastAPI REST API** — Real-time prediction endpoint for easy integration  
- 🐳 **Dockerized Deployment** — Full containerization for portability and scalability  
- 🔁 **CI/CD Pipeline (GitHub Actions + Render)** — Automated testing, building, and deployment  
- 📊 **Monitoring** — Prometheus + Grafana for system metrics, MLflow for model tracking  
- 🪵 **Logging** — Centralized request and error logging  

---

## Project Architecture
```
credit-risk-system/
│
├── src/
│ ├── main.py                     # FastAPI app entry point
│ ├── featuring.py                # Model training script
│ ├── logger.py                   # Logging configuration
│ ├── model_train.py              # Model train 
| ├── preprocess.py               # Data preprocessing
│
├── app/                          # FastAPI files
├── Streamlit_app                 # Streamlit web interfeys
├── Notebooks/                    # .Ipynb files
├── tests/                        # Unit tests (pytest)
├── data/                         # Datasets
├── models/                       # Saved model files
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Services (Prometheus, Grafana)
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
└── .github/workflows/ci.yml      # CI/CD pipeline config
```

---

## ⚡️ Tech Stack
| Category | Tools |
|-----------|--------|
| **Language** | Python 3.10 |
| **Frameworks** | FastAPI, Scikit-learn |
| **DevOps** | Docker, GitHub Actions, Render |
| **Monitoring** | Prometheus, Grafana, MLflow |
| **Testing** | Pytest |
| **Version Control** | Git + GitHub |

---

## 🧩 API Endpoints
| Method | Endpoint | Description |
|--------|-----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/predict` | Predict credit risk for new data |

Example request:
```bash
curl -X GET "https://credit-risk-system.onrender.com/predict?credit_score=720&income=4000"
```
Response:
```
{"approved": true}
```
# Run Locally
```
# Clone repository
git clone https://github.com/<FozilovDiyorbek>/credit-risk-system.git
cd credit-risk-system

# Build and run
docker build -t credit-risk-system .
docker run -p 8000:8000 credit-risk-system
```
Then open 👉 http://localhost:8000

## 📈 Model Training (MLflow Tracking)
```
python src/main.py
mlflow ui
```
# Author
- Diyorbek Fozilov
- Machine Learning Engineer
- 📧 [diyorbekfozilov011@gmail.com]
- 🌐 [LinkedIn](https://www.linkedin.com/in/diyorbek-fozilov-251975305/)

