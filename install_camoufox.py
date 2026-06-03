import logging
from camoufox import Camoufox

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_camoufox():
    """
    Initialize Camoufox to trigger the browser download.
    This script is intended to be run during the Docker build process.
    """
    logger.info("🚀 Starting Camoufox browser download...")
    try:
        # Initialize Camoufox - this triggers the download if not present
        browser = Camoufox(headless=True)
        logger.info("✅ Camoufox browser downloaded and installed successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to install Camoufox: {e}")
        raise

if __name__ == "__main__":
    install_camoufox()
