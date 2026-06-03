import os

dockerfile_path = 'Dockerfile'

with open(dockerfile_path, 'r') as f:
    lines = f.readlines()

new_lines = []
inserted_env = False

for line in lines:
    # Insert ENV before the install script
    if 'COPY install_camoufox.py .' in line and not inserted_env:
        new_lines.append('# Set cache directory for Camoufox to ensure it persists and is found at runtime\n')
        new_lines.append('ENV XDG_CACHE_HOME=/app/.cache\n')
        new_lines.append('RUN mkdir -p /app/.cache\n\n')
        inserted_env = True
    
    new_lines.append(line)

with open(dockerfile_path, 'w') as f:
    f.writelines(new_lines)

print("Dockerfile updated with XDG_CACHE_HOME successfully")
