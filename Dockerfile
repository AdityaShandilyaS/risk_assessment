FROM python:3.10
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY requirements.txt .
RUN pip install -r requirements.txt
ENTRYPOINT [ "python3" ]
CMD [ "risk_assessment/main.py" ]