import os

dockerfile_path = 'Dockerfile'

with open(dockerfile_path, 'r') as f:
    lines = f.readlines()

new_lines = []
inserted = False

for line in lines:
    new_lines.append(line)
    if 'echo "✅ Playwright browsers installed successfully"' in line and not inserted:
        new_lines.append('\n')
        new_lines.append('# Install Camoufox browser (pre-download to avoid runtime timeouts)\n')
        new_lines.append('COPY install_camoufox.py .\n')
        new_lines.append('RUN python3 install_camoufox.py && \\\n')
        new_lines.append('    echo "✅ Camoufox browser installed successfully"\n')
        inserted = True

with open(dockerfile_path, 'w') as f:
    f.writelines(new_lines)

print("Dockerfile updated successfully")
