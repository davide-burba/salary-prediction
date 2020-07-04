FROM tiangolo/uvicorn-gunicorn:python3.7

# Make directories suited to your application
#RUN mkdir -p /home/project/app
WORKDIR /app

# Copy contents from your local to your docker container
COPY . /app

# Copy and install requirements
#COPY requirements.txt /home/project/app
RUN pip install --no-cache-dir -r requirements.txt
