"""
WSGI entry point for production deployment.
Use this file with Gunicorn or other WSGI servers.

Example:
    gunicorn wsgi:app -w 4 -b 0.0.0.0:5000
"""

from app import app, load_all_breed_info, get_predictor

# Initialize on startup
load_all_breed_info()
get_predictor()

if __name__ == "__main__":
    app.run()
