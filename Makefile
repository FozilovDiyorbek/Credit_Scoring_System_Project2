install:
	pip install -r requirements.txt

test:
	pytest tests/

train:
	python src/main.py

run:
    uvicorn app.main:app --reload

docker-build:
	docker build -t credit-scoring-system .

docker-run:
	docker run -p 8000:8000 credit-scoring-system

streamlit:
	streamlit run Streamlit_app/app.py
	
