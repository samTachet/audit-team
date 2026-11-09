# Déploiement et retour arrière

```
./scripts/deploy.sh            # portes de qualité → build → up → santé → smoke ; retour auto si échec
./scripts/rollback.sh [tag]    # retour manuel (par défaut : version précédente enregistrée)
```

## Déroulé de `deploy.sh`

1. **Arbre git propre exigé** : l'image doit correspondre à un commit. Tag d'image = `git rev-parse --short=12 HEAD` (immuable, jamais `latest`).
2. **Portes de qualité** (`DEPLOY_GATE_CMD`, par défaut) : `pytest` → `spec_check` → `docs_gen --check` → `regress`. Un échec arrête tout avant le moindre appel Docker.
   `regress` rejoue les 18 scénarios de référence et compare à `baselines/reference.json` (comportement **et** empreinte prompt/modèle/tools).
3. `docker compose build api` puis `up -d --no-build api` avec `IMAGE_TAG=<sha>`.
4. Attente de l'état `healthy` (healthcheck compose sur `/healthz`), puis **smoke test** si `ACSI_SMOKE_KEY` est défini :
   santé, tâche nominale, **refus de politique effectif** (tool interdit), authentification exigée. Ces contrôles ne créent aucun effet de bord.
5. Succès : `.deploy/current` = nouveau tag, `.deploy/previous` = ancien. Échec : la version précédente est relancée **avant** toute mise à jour de l'état,
   donc `current` ne désigne jamais une version malsaine.

| Code | Sens |
|---|---|
| 0 | déployé |
| 1 | refusé avant déploiement (arbre sale, portes de qualité) |
| 2 | déploiement échoué, version précédente rétablie (ou service arrêté s'il n'y en avait pas) |
| 3 | échec du retour arrière lui-même : intervention manuelle |

`rollback.sh` vérifie que l'image cible existe localement avant de toucher quoi que ce soit, valide le format du tag
(`[A-Za-z0-9_.-]+`) et rejoue santé + smoke.

## Données et compatibilité

Le volume `acsi-journal` survit aux versions. Pour qu'un retour arrière reste possible :

- le format du journal ne doit évoluer que de façon **additive** (nouveaux champs ou nouveaux types d'événements ; le décodage ignore les types inconnus) ;
- une évolution **non additive** exige un instantané du volume avant déploiement : revenir à une version antérieure ne restaure pas les données.

## audit-team : un seul worker, trois volumes

L'API `audit_team.api:create_app` refuse de démarrer si `WEB_CONCURRENCY`/`UVICORN_WORKERS` > 1. `compose.yaml` monte
trois volumes : `acsi-journal` (journal), `acsi-anchors` (ancres de tête, séparées pour qu'une troncature du journal
seul soit détectée) et `acsi-state` (registre d'idempotence et checkpoints LangGraph SQLite). La clé
`ACSI_JOURNAL_HMAC_KEY` doit venir du gestionnaire de secrets, pas d'un de ces volumes. Un retour arrière vers la v1
(`acsi-agent`) n'est pas prévu : les journaux v2 portent des champs additionnels (`actor`, `node`, `checkpoint_id`, `mac`).

## Ce qui est vérifié, et ce qui ne l'est pas

- Vérifié par `tests/test_deploy.py` avec un faux `docker` : ordre des opérations, état `current`/`previous`, retour automatique, refus sur arbre sale / portes en échec / tag malformé / image absente.
- **Non vérifié en local** : la construction réelle de l'image et le démarrage du conteneur (pas de démon Docker dans l'environnement de développement). Le job `docker` de `.github/workflows/ci.yml` les exécute (build, `docker compose run --rm eval`, `/healthz` de l'API).
- **CI GitHub exécutée** : run 42717155158 du 2026-11-10 00:01 (heure de Paris) sur le commit `43db11d`, verte sur les trois jobs :
  `verify` (spec_check, docs_gen --check, pytest 284 passés, regress, `audit_team.eval --runner crewai`, demo),
  `dependencies` (pip-audit : aucune vulnérabilité hors les 4 avis ChromaDB ignorés et justifiés) et `docker`
  (build de l'image, `docker compose run --rm eval` réseau coupé + FS en lecture seule, API un worker répondant sur `/healthz`).
  Le premier run (42716947773) avait échoué au build Docker (README.md absent de l'image) ; corrigé dans `43db11d`.
- Non exécuté : `scripts/deploy.sh` contre un vrai démon Docker, et `scripts/smoke.py` contre l'API conteneurisée.
