"""WSGI entrypoint so VoiceShield can be launched via `gunicorn wsgi:app`."""

from app.main import app as fastapi_app

# Gunicorn looks for a module-level variable named `app`
app = fastapi_app
