"""TruthLens Streamlit frontend package.

Explicitly a regular package rather than a namespace package: the project root
and ``app/`` are both on ``sys.path`` when Streamlit runs ``app/app.py``, and
without this file ``import app`` resolves to ``app/app.py`` as a top-level
module instead of to this directory — which breaks ``from app import theme``
and re-executes the page.
"""
