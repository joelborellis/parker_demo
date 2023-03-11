# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

WORKDIR /

COPY . .

RUN pip3 install -r requirements.txt

CMD streamlit run app.py
