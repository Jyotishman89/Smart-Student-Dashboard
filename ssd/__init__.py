"""Smart Student Dashboard — a multi-user academic tracking app.

Package layout:
    config      constants and runtime settings (secrets/env)
    db          SQLAlchemy engine + session factory (Postgres, SQLite fallback)
    models      ORM models
    academics   pure calculation logic (no Streamlit / DB imports)
    repository  data-access layer (CRUD)
    auth        signup / login / session handling
    charts      Plotly figure builders
    theme       page config + CSS
    main        app entrypoint (auth gate + navigation)
    views/      one module per page
"""

__version__ = "2.0.0"
