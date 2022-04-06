FROM python:slim

WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN python -m spacy download en

COPY . .

CMD ["python", "TODO_analyses.py"]