# Dockerfile for Universal Scraper API - Cloud Run
FROM python:3.11-slim

# Set working directory
WORKDIR /app
# Set cache directory for browsers to ensure they persist and are found at runtime
ENV XDG_CACHE_HOME=/app/.cache
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.cache/ms-playwright
RUN mkdir -p /app/.cache


# Install system dependencies (including Tesseract for OCR and browser dependencies)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    tesseract-ocr \
    tesseract-ocr-eng \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libwayland-client0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (required for browser automation)
# Skip verification to speed up build - verification happens at runtime
RUN playwright install chromium && \
    playwright install-deps chromium && \
    echo "✅ Playwright browsers installed successfully"

# Install Camoufox browser (pre-download to avoid runtime timeouts)
# Set cache directory for Camoufox to ensure it persists and is found at runtime

# Fetch Camoufox binaries using the CLI (respects XDG_CACHE_HOME)
RUN python3 -m camoufox fetch \
    && echo "✅ Camoufox browser installed successfully" \
    || echo "⚠️ Camoufox fetch failed — will attempt at runtime"

# Copy application code
COPY universal_scraper/ ./universal_scraper/
COPY api/ ./api/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Create a startup script that ensures the server starts quickly and handles read-only filesystem
RUN echo '#!/bin/sh\n\
set -e\n\
\n\
# Handle Cloud Run read-only filesystem\n\
# We create writable directories in /tmp and symlink the read-only binaries into them\n\
echo "Configuring writable cache for Cloud Run..."\n\
mkdir -p /tmp/cache/ms-playwright\n\
mkdir -p /tmp/cache/camoufox\n\
\n\
# Symlink Playwright browser contents (binaries are read-only, but directory must be writable for locks)\n\
if [ -d "/app/.cache/ms-playwright" ]; then\n\
    echo "Symlinking Playwright browsers..."\n\
    for item in /app/.cache/ms-playwright/*; do\n\
        [ -e "$item" ] || continue\n\
        ln -sf "$item" "/tmp/cache/ms-playwright/$(basename "$item")"\n\
    done\n\
fi\n\
\n\
# Symlink Camoufox contents\n\
if [ -d "/app/.cache/camoufox" ]; then\n\
    echo "Symlinking Camoufox binaries..."\n\
    for item in /app/.cache/camoufox/*; do\n\
        [ -e "$item" ] || continue\n\
        ln -sf "$item" "/tmp/cache/camoufox/$(basename "$item")"\n\
    done\n\
fi\n\
\n\
# Set environment variables to point to writable /tmp/cache\n\
export XDG_CACHE_HOME=/tmp/cache\n\
export PLAYWRIGHT_BROWSERS_PATH=/tmp/cache/ms-playwright\n\
\n\
echo "Starting Universal Scraper API on port ${PORT:-8080}..."\n\
exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --log-level info --access-log\n\
' > /app/start.sh && chmod +x /app/start.sh

# Start the server
CMD ["/app/start.sh"]
