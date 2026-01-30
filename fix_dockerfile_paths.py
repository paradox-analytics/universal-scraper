import os

dockerfile_path = 'Dockerfile'

with open(dockerfile_path, 'r') as f:
    content = f.read()

# Define the new environment variables and directory creation
env_setup = """
# Set cache directory for browsers to ensure they persist and are found at runtime
ENV XDG_CACHE_HOME=/app/.cache
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.cache/ms-playwright
RUN mkdir -p /app/.cache
"""

# Remove the old ENV and mkdir lines if they exist
content = content.replace('ENV XDG_CACHE_HOME=/app/.cache\n', '')
content = content.replace('RUN mkdir -p /app/.cache\n', '')

# Insert the new setup after WORKDIR /app
if 'WORKDIR /app' in content:
    content = content.replace('WORKDIR /app', 'WORKDIR /app' + env_setup)

# Ensure playwright install and camoufox fetch are in the right place
# (They already are after the requirements install, so just moving the ENV up is enough)

with open(dockerfile_path, 'w') as f:
    f.write(content)

print("Dockerfile updated with correct environment variable order.")
