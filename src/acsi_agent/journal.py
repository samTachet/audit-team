"""Journal d'événements append-only, un fichier JSONL par requête ou mission, chaîné par hachage.

Chaque événement porte `prev` (hachage du précédent) et `hash`. En plus de la chaîne (v1) :

- **tête ancrée hors du fichier** (I1) : après chaque ajout, `(seq, hash)` du dernier événement est écrit
  dans un répertoire d'ancres séparé (`anchor_dir`, par défaut `<journal>-anchors`, configurable par
  `ACSI_ANCHOR_DIR` pour le placer hors du volume du journal). `verify` compare la fin du fichier à l'ancre :
  une troncature ou une réécriture complète avec chaîne recalculée est détectée ;
- **HMAC optionnel** : si une clé est fournie (`ACSI_JOURNAL_HMAC_KEY`, à garder hors du volume), chaque
  événement et chaque ancre portent un `mac` = HMAC-SHA256(clé, hash). Sans clé, AUCUNE authenticité n'est
  garantie : quelqu'un qui peut écrire à la fois le journal et les ancres peut tout réécrire ;
- **scellement** : `seal()` marque l'ancre comme scellée ; tout ajout ultérieur est refusé ;
- tête gardée en mémoire : l'ajout ne relit plus tout le fichier (N3), verrou de fichier `fcntl` (I5).
"""
from __future__ import annotations

import fcntl
import hashlib
import hmac as _hmac
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .types import REQUEST_ID_RE

GENESIS = "0" * 64


class JournalSealed(Exception):
    """Ajout refusé : le journal de cette requête est scellé."""


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class Journal:
    def __init__(self, directory: Path, anchor_dir: Path | None = None, hmac_key: bytes | None = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        env_anchor = os.environ.get("ACSI_ANCHOR_DIR")
        self.anchor_dir = Path(anchor_dir or env_anchor or self.dir.with_name(self.dir.name + "-anchors"))
        self.anchor_dir.mkdir(parents=True, exist_ok=True)
        self.hmac_key = hmac_key
        self._lock = threading.Lock()
        self._heads: dict[str, tuple[int, str]] = {}

    # ------------------------------------------------------------ chemins
    def path(self, request_id: str) -> Path:
        # Validation stricte : l'identifiant ne doit jamais pouvoir désigner un autre chemin.
        if not REQUEST_ID_RE.fullmatch(request_id):
            raise ValueError("request_id invalide")
        return self.dir / f"{request_id}.jsonl"

    def _anchor_path(self, request_id: str) -> Path:
        self.path(request_id)  # même validation
        return self.anchor_dir / f"{request_id}.head.json"

    def _mac(self, h: str) -> str | None:
        return _hmac.new(self.hmac_key, h.encode(), hashlib.sha256).hexdigest() if self.hmac_key else None

    # ------------------------------------------------------------ lecture
    def read(self, request_id: str) -> list[dict[str, Any]]:
        p = self.path(request_id)
        if not p.exists():
            return []
        with p.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def anchor(self, request_id: str) -> dict[str, Any] | None:
        p = self._anchor_path(request_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def _write_anchor(self, request_id: str, seq: int, h: str, sealed: bool = False) -> None:
        data = {"seq": seq, "hash": h, "sealed": sealed}
        if self.hmac_key:
            data["mac"] = self._mac(f"{seq}:{h}:{int(sealed)}")
        tmp = self._anchor_path(request_id).with_suffix(".tmp")
        tmp.write_text(_canon(data), encoding="utf-8")
        os.replace(tmp, self._anchor_path(request_id))

    def _head(self, request_id: str) -> tuple[int, str]:
        if request_id not in self._heads:
            events = self.read(request_id)
            self._heads[request_id] = (len(events), events[-1]["hash"] if events else GENESIS)
        return self._heads[request_id]

    def head(self, request_id: str) -> tuple[int, str]:
        with self._lock:
            return self._head(request_id)

    # ------------------------------------------------------------ écriture
    def append(self, request_id: str, type_: str, data: dict[str, Any], ts: float | None = None,
               meta: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            anc = self.anchor(request_id)
            if anc and anc.get("sealed"):
                raise JournalSealed(f"journal {request_id} scellé : ajout refusé")
            seq, prev = self._head(request_id)
            core = {"seq": seq + 1, "ts": ts if ts is not None else time.time(),
                    "request_id": request_id, "type": type_, "data": data, "prev": prev}
            if meta:
                core.update({k: v for k, v in meta.items() if v is not None})
            h = hashlib.sha256((prev + _canon(core)).encode()).hexdigest()
            event = {**core, "hash": h}
            if self.hmac_key:
                event["mac"] = self._mac(h)
            with self.path(request_id).open("a", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # verrou de fichier : un seul écrivain à la fois
                try:
                    f.write(_canon(event) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            self._heads[request_id] = (seq + 1, h)
            self._write_anchor(request_id, seq + 1, h)
            return event

    def seal(self, request_id: str) -> None:
        with self._lock:
            seq, h = self._head(request_id)
            self._write_anchor(request_id, seq, h, sealed=True)

    def forget(self, request_id: str) -> None:
        """Oublie la tête en cache (tests : simule un redémarrage du processus)."""
        with self._lock:
            self._heads.pop(request_id, None)

    # ------------------------------------------------------------ vérification
    def verify(self, request_id: str) -> tuple[bool, str]:
        prev = GENESIS
        events = self.read(request_id)
        for i, e in enumerate(events, start=1):
            core = {k: v for k, v in e.items() if k not in {"hash", "mac"}}
            if e.get("seq") != i or e.get("prev") != prev:
                return False, f"chaîne rompue à l'événement {i}"
            if hashlib.sha256((prev + _canon(core)).encode()).hexdigest() != e.get("hash"):
                return False, f"contenu altéré à l'événement {i}"
            if self.hmac_key and not _hmac.compare_digest(str(e.get("mac", "")), self._mac(e["hash"]) or ""):
                return False, f"HMAC invalide à l'événement {i}"
            prev = e["hash"]
        anc = self.anchor(request_id)
        if anc is None:
            return (False, "tête non ancrée") if events else (True, "ok")
        if self.hmac_key:
            expect = self._mac(f"{anc.get('seq')}:{anc.get('hash')}:{int(bool(anc.get('sealed')))}")
            if not _hmac.compare_digest(str(anc.get("mac", "")), expect or ""):
                return False, "ancre de tête altérée (HMAC)"
        if len(events) < anc["seq"]:
            return False, f"journal tronqué : {len(events)} événement(s) pour une tête ancrée à {anc['seq']}"
        if len(events) > anc["seq"]:
            return False, f"événements non ancrés après {anc['seq']}"
        if prev != anc["hash"]:
            return False, "dernier événement différent de la tête ancrée"
        return True, "ok"

    def list_ids(self, prefix: str = "req_") -> list[str]:
        return sorted(p.stem for p in self.dir.glob(f"{prefix}*.jsonl") if REQUEST_ID_RE.fullmatch(p.stem))
