FROM ubuntu:14.04
EXPOSE 4000
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential libglib2.0-0
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip==9.0.1
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["src/app.py"]
