import os

dockerfile_path = 'Dockerfile'

with open(dockerfile_path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'COPY install_camoufox.py .' in line:
        continue  # Skip copying the script
    elif 'RUN python3 install_camoufox.py' in line:
        new_lines.append('# Fetch Camoufox binaries using the CLI (respects XDG_CACHE_HOME)\n')
        new_lines.append('RUN python3 -m camoufox fetch && \\\n')
    else:
        new_lines.append(line)

with open(dockerfile_path, 'w') as f:
    f.writelines(new_lines)

print("Dockerfile updated with camoufox fetch successfully")
