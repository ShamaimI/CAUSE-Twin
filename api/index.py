import sys
import os

# Ensure the root directory and webapp directory are in the path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'webapp'))

from webapp.app import app

# Vercel serverless WSGI handler
def handler(request, response):
    return app(request, response)