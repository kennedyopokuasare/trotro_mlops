# Base image from https://hub.docker.com/_/python
FROM python:3.10-slim

# upgrade pip
RUN pip install --upgrade pip

# create working direcotry
WORKDIR /app

# copy over deployment files
RUN mkdir -p model/

COPY model/ model/

COPY predict.py requirements.txt ./

# install other requirements
RUN pip install -r requirements.txt

# install model dependencies with requirements
RUN pip install -r model/requirements.txt

# Open port for out of container requests
EXPOSE 9696

# main command/ entry point starts gunicorn WSGI server
ENTRYPOINT [ "gunicorn",  "--bind=0.0.0.0:9696", "predict:app" ]
