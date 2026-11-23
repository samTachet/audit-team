FROM python:3.13-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=0
RUN pip install --no-cache-dir uv==0.11.32
WORKDIR /app

# Dépendances d'abord (couche mise en cache tant que pyproject/uv.lock ne changent pas)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md LICENSE ./
COPY src ./src
COPY specs ./specs
COPY scenarios ./scenarios
COPY data/fixtures.json ./data/fixtures.json
RUN uv sync --frozen --no-dev

# Utilisateur sans privilèges ; seuls journal, ancres et état (idempotence, checkpoints) sont inscriptibles.
RUN useradd --system --uid 10001 --no-create-home acsi \
 && mkdir -p /app/data/journal /app/data/anchors /app/data/state \
 && chown -R acsi /app/data/journal /app/data/anchors /app/data/state
USER acsi
# HOME=/tmp : CrewAI écrit quelques fichiers locaux (~/.config/crewai, ~/.local/share) ; /tmp est un tmpfs.
# Télémétrie CrewAI/OpenTelemetry coupée.
ENV PATH="/app/.venv/bin:$PATH" HOME=/tmp ACSI_ROOT=/app ACSI_JOURNAL_DIR=/app/data/journal ACSI_BACKEND=mock \
    CREWAI_DISABLE_TELEMETRY=true CREWAI_DISABLE_TRACKING=true CREWAI_TRACING_ENABLED=false OTEL_SDK_DISABLED=true

# Par défaut : évaluation hermétique v1 (S01–S18) puis équipe (M01…M13, runner CrewAI, doubles déterministes)
CMD ["sh", "-c", "python -m acsi_agent.eval && python -m audit_team.eval"]
