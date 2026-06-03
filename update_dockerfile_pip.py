import os

dockerfile_path = 'Dockerfile'

with open(dockerfile_path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'RUN pip install --no-cache-dir -r requirements.txt' in line:
        new_lines.append('# Upgrade pip first\n')
        new_lines.append('RUN pip install --no-cache-dir --upgrade pip && \\\n')
        new_lines.append('    pip install --no-cache-dir -r requirements.txt\n')
    else:
        new_lines.append(line)

with open(dockerfile_path, 'w') as f:
    f.writelines(new_lines)

print("Dockerfile updated with pip upgrade successfully")
