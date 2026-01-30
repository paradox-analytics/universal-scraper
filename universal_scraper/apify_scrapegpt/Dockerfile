# Universal Web Scraper - Apify Actor Dockerfile
# Supports crawling, scraping, and AI-powered extraction

# Use Playwright base image (includes Playwright + Chromium + all browser dependencies)
FROM apify/actor-python-playwright:3.11

# Set working directory
WORKDIR /usr/src/app

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional system dependencies for Camoufox (Firefox-based browser)
# Playwright image already has most dependencies, we only need Firefox-specific ones
RUN apt-get update && apt-get install -y \
    libgtk-3-0 \
    libgtk-3-dev \
    libdbus-glib-1-2 \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# Install Camoufox (advanced anti-detection browser)
RUN pip install camoufox>=0.4.0 camoufox[geoip]

# Copy pre-download script
COPY pre_download_camoufox.py ./

# Pre-download Camoufox browser to avoid 713MB download on every run
# This downloads the browser binary to /root/.cache/camoufox during build
RUN python3 pre_download_camoufox.py

# Copy the entire universal_scraper package
COPY core ./universal_scraper/core
COPY crawler ./universal_scraper/crawler
COPY orchestrator ./universal_scraper/orchestrator
COPY __init__.py ./universal_scraper/__init__.py

# Copy actor files (main.py is the entry point with Web Unblocker support)
COPY main.py ./

# Set Python path
ENV PYTHONPATH=/usr/src/app:$PYTHONPATH

# Run the actor
CMD ["python", "main.py"]
