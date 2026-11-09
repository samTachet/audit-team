"""Implémentations des tools sur les services simulés.

`classify_excerpt` est un substitut déterministe à base de mots-clés, au même contrat que
`classify()` de acsi-audit-classify : il sera rebranché sur ce module sans changer la spec.
"""
from __future__ import annotations

import hashlib
import unicodedata
from typing import Any, Callable

from .backends import ExternalUnavailable, Services
from .idempotency import IdempotencyRegistry, idempotency_key
from .types import CallContext


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s.casefold()).encode("ascii", "ignore").decode()


# (règlement, article, gravité, mots-clés normalisés) — ordre = priorité en cas d'égalité
RULES: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("AI_ACT", "10", "high", ("biais", "gouvernance des donnees", "jeux de donnees d'entrainement")),
    ("AI_ACT", "14", "high", ("supervision humaine", "controle humain", "intervention humaine")),
    ("AI_ACT", "11", "medium", ("documentation technique",)),
    ("AI_ACT", "9", "medium", ("gestion des risques",)),
    ("RGPD", "32", "high", ("chiffrement", "securite du traitement", "acces non autorise", "mots de passe")),
    ("RGPD", "33", "high", ("violation de donnees", "notification a la cnil", "72 heures")),
    ("RGPD", "35", "high", ("analyse d'impact", "aipd", "dpia")),
    ("RGPD", "13", "medium", ("information des personnes", "mentions d'information", "politique de confidentialite")),
    ("RGPD", "6", "medium", ("consentement", "base legale")),
]


def classify(text: str) -> dict[str, Any]:
    t = _norm(text)
    best: tuple[int, tuple[str, str, str], list[str]] | None = None
    for reg, art, sev, kws in RULES:
        hits = [k for k in kws if k in t]
        if hits and (best is None or len(hits) > best[0]):
            best = (len(hits), (reg, art, sev), hits)
    if best is None:
        return {"regulation": "NONE", "article": None, "severity": "none", "matched_keywords": []}
    (reg, art, sev), hits = best[1], best[2]
    return {"regulation": reg, "article": art, "severity": sev, "matched_keywords": hits}


def _short_id(prefix: str, *parts: object) -> str:
    return prefix + hashlib.sha1(":".join(map(str, parts)).encode()).hexdigest()[:6]


def build_impls(svc: Services, registry: IdempotencyRegistry | None = None
                ) -> dict[str, Callable[[dict[str, Any], CallContext], dict[str, Any]]]:
    def classify_excerpt(args, ctx):
        return classify(args["text"])

    def lookup_article(args, ctx):
        entry = svc.corpus.get(args["regulation"], {}).get(args["article"])
        if entry is None:
            raise ToolError("ARTICLE_NOT_FOUND", f"{args['regulation']} art. {args['article']} absent du corpus local")
        return {"regulation": args["regulation"], "article": args["article"], **entry}

    def list_nonconformities(args, ctx):
        d = svc.dossiers.get(args["dossier_id"])
        if d is None:
            raise ToolError("DOSSIER_NOT_FOUND", f"dossier {args['dossier_id']} introuvable")
        return {"dossier_id": args["dossier_id"], "client": d["client"], "nonconformities": [dict(n) for n in d["nonconformities"]]}

    def draft_finding(args, ctx):
        d = svc.dossiers.get(args["dossier_id"])
        if d is None:
            raise ToolError("DOSSIER_NOT_FOUND", f"dossier {args['dossier_id']} introuvable")
        if args["nc_id"] not in {n["id"] for n in d["nonconformities"]}:
            raise ToolError("NC_NOT_FOUND", f"{args['nc_id']} n'appartient pas à {args['dossier_id']}")
        draft_id = _short_id("DR-", ctx.idem_key) if ctx.idem_key else _short_id("DR-", ctx.request_id, ctx.step)
        if ctx.idem_key and draft_id in svc.drafts:  # reprise ou nouveau tour de revue : pas de doublon
            dr = svc.drafts[draft_id]
            if dr["status"] == "draft" and dr["text"] != args["text"]:
                dr["text"] = args["text"]  # révision du même brouillon (effet réversible)
            return {"draft_id": draft_id, "dossier_id": dr["dossier_id"], "nc_id": dr["nc_id"], "status": dr["status"]}
        svc.drafts[draft_id] = {"dossier_id": args["dossier_id"], "nc_id": args["nc_id"], "text": args["text"], "status": "draft"}
        svc.drafts_created += 1
        return {"draft_id": draft_id, "dossier_id": args["dossier_id"], "nc_id": args["nc_id"], "status": "draft"}

    def delete_draft(args, ctx):
        dr = svc.drafts.get(args["draft_id"])
        if dr is None:
            raise ToolError("DRAFT_NOT_FOUND", f"brouillon {args['draft_id']} introuvable")
        if dr["status"] == "submitted":
            raise ToolError("DRAFT_SUBMITTED", "brouillon déjà transmis")
        del svc.drafts[args["draft_id"]]
        svc.drafts_deleted += 1
        return {"draft_id": args["draft_id"], "status": "deleted"}

    def submit_dossier(args, ctx):
        d = svc.dossiers.get(args["dossier_id"])
        if d is None:
            raise ToolError("DOSSIER_NOT_FOUND", f"dossier {args['dossier_id']} introuvable")
        if d["status"] == "submitted":
            raise ToolError("ALREADY_SUBMITTED", "dossier déjà transmis")
        ids = sorted(i for i, dr in svc.drafts.items() if dr["dossier_id"] == args["dossier_id"] and dr["status"] == "draft")
        if not ids:
            raise ToolError("NO_DRAFTS", "aucun brouillon à transmettre")
        texts = [svc.drafts[i]["text"] for i in ids]
        body = "\n\n".join(texts)
        key = idempotency_key(args["dossier_id"], args["recipient"], texts)
        if registry is not None:  # I3 : idempotence persistante entre tâches et missions
            ok, cur = registry.reserve(args["dossier_id"], key, ctx.request_id)
            if not ok:
                if cur and cur.get("state") == "submitted":
                    raise ToolError("ALREADY_SUBMITTED", "dossier déjà transmis (registre d'idempotence)")
                raise ToolError("OUTCOME_UNKNOWN", "un envoi précédent a une issue indéterminée : rapprochement requis")
        try:
            svc.mail.send(args["recipient"], f"Dossier d'audit {args['dossier_id']}", body)
        except ExternalUnavailable as e:
            if registry is not None:
                registry.release(args["dossier_id"], key)
            raise ToolError("EXTERNAL_UNAVAILABLE", str(e)) from e
        for i in ids:
            svc.drafts[i]["status"] = "submitted"
        d["status"] = "submitted"
        sub_id = _short_id("SUB-", ctx.request_id, ctx.step)
        if registry is not None:
            registry.complete(args["dossier_id"], key, sub_id)
        svc.submissions.append({"submission_id": sub_id, "dossier_id": args["dossier_id"], "recipient": args["recipient"]})
        return {"submission_id": sub_id, "dossier_id": args["dossier_id"], "recipient": args["recipient"],
                "drafts_submitted": len(ids), "status": "submitted"}

    return {
        "classify_excerpt": classify_excerpt,
        "lookup_article": lookup_article,
        "list_nonconformities": list_nonconformities,
        "draft_finding": draft_finding,
        "delete_draft": delete_draft,
        "submit_dossier": submit_dossier,
    }
