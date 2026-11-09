"""Registre d'idempotence persistant des envois (I3 / EF-11).

Les services simulés sont recréés à chaque tâche : sans état partagé, `ALREADY_SUBMITTED` ne protégeait
qu'à l'intérieur d'une tâche. Ce registre, persistant (fichier JSON, verrou `fcntl`), est consulté par
`submit_dossier` AVANT tout envoi, quelle que soit la tâche ou la mission :

- `submitted` : le dossier a été transmis → toute nouvelle tentative reçoit `ALREADY_SUBMITTED` ;
- `in_flight` : un envoi a commencé et son issue n'est pas connue (timeout, crash) → `OUTCOME_UNKNOWN`
  bloque toute nouvelle tentative jusqu'au rapprochement manuel (`reconcile`).

Clé d'idempotence = sha256(dossier, destinataire normalisé, hachage des brouillons) ; elle est enregistrée
avec l'issue. Le verrou est par dossier : un même dossier ne part qu'une fois, même vers un autre destinataire.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def idempotency_key(dossier_id: str, recipient: str, drafts_text: list[str]) -> str:
    h = hashlib.sha256()
    h.update(dossier_id.encode())
    h.update(b"\x00" + recipient.strip().lower().encode())
    for t in drafts_text:
        h.update(b"\x00" + hashlib.sha256(t.encode()).digest())
    return h.hexdigest()


class IdempotencyRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        with self._lock:
            lock_path = self.path.with_suffix(".lock")
            with lock_path.open("a+") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
                    yield data
                    tmp = self.path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
                    os.replace(tmp, self.path)
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def get(self, dossier_id: str) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8")).get(dossier_id)

    def reserve(self, dossier_id: str, key: str, by: str) -> tuple[bool, dict[str, Any] | None]:
        """Réserve l'envoi du dossier. Renvoie (True, None) si l'envoi peut partir, sinon (False, entrée existante)."""
        with self._locked() as data:
            cur = data.get(dossier_id)
            if cur is not None:
                return False, dict(cur)
            data[dossier_id] = {"state": "in_flight", "key": key, "by": by, "ts": time.time()}
            return True, None

    def complete(self, dossier_id: str, key: str, submission_id: str) -> None:
        with self._locked() as data:
            data[dossier_id] = {**data.get(dossier_id, {}), "state": "submitted", "key": key,
                                "submission_id": submission_id, "ts": time.time()}

    def release(self, dossier_id: str, key: str) -> None:
        """Échec certain (le service n'a rien envoyé) : la réservation est levée."""
        with self._locked() as data:
            if data.get(dossier_id, {}).get("key") == key and data[dossier_id]["state"] == "in_flight":
                del data[dossier_id]

    def reconcile(self, dossier_id: str, sent: bool, note: str = "") -> None:
        """Rapprochement opérateur d'une issue indéterminée : marque transmis ou libère."""
        with self._locked() as data:
            if dossier_id not in data:
                return
            if sent:
                data[dossier_id].update(state="submitted", reconciled=note or True)
            else:
                del data[dossier_id]
