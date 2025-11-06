# Use official Apify base image
FROM apify/actor-python:3.11

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy scraper code
COPY universal_scraper ./universal_scraper/

# Copy Apify files
COPY universal_scraper/apify/actor.py ./
COPY universal_scraper/apify/INPUT_SCHEMA.json ./.actor/INPUT_SCHEMA.json

# Set entrypoint
CMD ["python", "-m", "universal_scraper.apify.actor"]

