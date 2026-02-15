FROM python:3.6-stretch

WORKDIR /tmp
COPY ./requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

WORKDIR /app
COPY ./start.sh /home/start.sh
CMD ["/home/start.sh"]
