"""audit-team : équipe d'agents d'audit (CrewAI) orchestrée par un graphe (LangGraph), derrière la gateway ACSI.

Les frameworks sont des locataires de la couche ACSI (`acsi_agent`) : ce paquet n'importe ni `backends` ni
`tools_impl` (test d'architecture) et n'atteint les tools que via `TeamRuntime.propose` / `execute_approved`.
"""
import os

# ENF-07 : télémétrie et traçage distants de CrewAI coupés AVANT tout import de crewai (désactivable
# explicitement en exportant ces variables avec une autre valeur).
for _var, _val in (("CREWAI_DISABLE_TELEMETRY", "true"), ("CREWAI_DISABLE_TRACKING", "true"),
                   ("OTEL_SDK_DISABLED", "true"), ("CREWAI_TRACING_ENABLED", "false"),
                   ("LANGSMITH_TRACING", "false"), ("LANGCHAIN_TRACING_V2", "false")):
    os.environ.setdefault(_var, _val)

__version__ = "2.0.0"
