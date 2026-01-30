#!/usr/bin/env python3
"""Pre-download Camoufox browser during Docker build"""

import sys
import os
import subprocess

# Check GTK3 libraries before starting
print(" Checking GTK3 libraries...")
try:
    result = subprocess.run(
        ["ldconfig", "-p"],
        capture_output=True,
        text=True
    )
    if "libgtk-3.so.0" in result.stdout:
        print(" libgtk-3.so.0 found")
    else:
        print(" libgtk-3.so.0 NOT found in library cache")
        print("Running ldconfig to update cache...")
        subprocess.run(["ldconfig"], check=False)
except Exception as e:
    print(f" Could not check libraries: {e}")

try:
    from camoufox.sync_api import Camoufox
    
    print(" Initializing Camoufox (this will download ~713MB browser)...")
    print(" Pre-downloading GeoIP database (this will download ~63MB database)...")
    
    # Initialize with geoip=True to pre-download GeoIP database during build
    # This ensures the database is cached in the Docker image
    browser = Camoufox(headless=True, geoip=True)
    print(" Entering browser context...")
    context = browser.__enter__()
    print(" Camoufox initialized successfully")
    
    # Verify cache location
    cache_dir = os.path.expanduser("~/.cache/camoufox")
    if os.path.exists(cache_dir):
        print(f" Camoufox cached at {cache_dir}")
        # List contents
        files = os.listdir(cache_dir)
        print(f"   Cache contains {len(files)} files/directories")
    else:
        print(f" Cache directory not found at {cache_dir}")
    
    # Verify GeoIP database is downloaded
    # The geoip2 library downloads the database to ~/.cache/geoip2/GeoLite2-City.mmdb
    # when Camoufox initializes with geoip=True
    geoip_cache_dir = os.path.expanduser("~/.cache/geoip2")
    if os.path.exists(geoip_cache_dir):
        print(f" GeoIP cache directory exists: {geoip_cache_dir}")
        geoip_files = os.listdir(geoip_cache_dir)
        print(f"   GeoIP cache contains {len(geoip_files)} files")
        for f in geoip_files:
            file_path = os.path.join(geoip_cache_dir, f)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   - {f}: {size_mb:.1f} MB")
    else:
        print(f" GeoIP cache directory not found at {geoip_cache_dir}")
        print("   (This is OK - it will be created on first use)")
    
    browser.__exit__(None, None, None)
    print(" Pre-download complete!")
    print("   Both Camoufox browser and GeoIP database are now cached in the Docker image")
except Exception as e:
    print(f" Camoufox pre-download failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

