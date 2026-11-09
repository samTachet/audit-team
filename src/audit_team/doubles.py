"""Doubles LLM déterministes, un par rôle (EF-16) : aucune clé, aucun réseau, sorties scriptées.

Un « cerveau » est une fonction pure de (entrées de la phase, issues des tools déjà rendues par la gateway) vers
la prochaine décision : `Act(tool, args)` ou `Final(sortie structurée)`. Le même cerveau sert :
- au runner direct (P1, sans framework) ;
- au runner CrewAI (P3), où `GovernedLLM` le traduit en texte ReAct (`Action:` / `Final Answer:`).

Les modes simulent des comportements de modèle (maladroit, trompé, emballé, en panne) pour éprouver la
plomberie ; la sécurité ne repose JAMAIS sur eux.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
INJECTION_RE = re.compile(r"ignore (tes|vos|les) consignes|sans demander", re.I)


@dataclass(frozen=True)
class Act:
    tool: str
    args: dict[str, Any]
    thought: str = ""


@dataclass(frozen=True)
class Final:
    output: dict[str, Any]
    thought: str = ""


class Brain(Protocol):
    id: str

    def next(self, inputs: dict[str, Any], outcomes: list[dict[str, Any]]) -> Act | Final: ...


class DoubleError(RuntimeError):
    """Panne simulée du modèle d'un rôle (M13)."""


def _ok(o: dict[str, Any]) -> bool:
    return o.get("status") == "ok"


# ------------------------------------------------------------------ analyste
@dataclass
class AnalystDouble:
    mode: str = "normal"  # normal | injected | raise
    id: str = "analyst-double-v1"

    def next(self, inputs, outcomes):
        if self.mode == "raise":
            raise DoubleError("le modèle de l'analyste ne répond pas")
        d = inputs["dossier_id"]
        if not outcomes:
            return Act("list_nonconformities", {"dossier_id": d}, "Je liste les non-conformités du dossier.")
        first = outcomes[0]
        if not _ok(first):
            return Final({"dossier_id": d, "error": first.get("code") or "LIST_FAILED"}, "Dossier illisible.")
        ncs = [n for n in first["result"]["nonconformities"] if n["status"] == "open"]
        plan: list[Act] = []
        hint = None
        notes = []
        for n in ncs:
            if INJECTION_RE.search(n["description"]):
                notes.append(f"{n['id']} contient une instruction dans les données : ignorée")
                if self.mode == "injected":  # modèle trompé : il obéit à l'instruction cachée
                    plan.append(Act("shell_exec", {"command": "cat /etc/passwd"}, "La note demande une commande."))
                    m = EMAIL_RE.search(n["description"])
                    hint = m.group(0) if m else None
        seen = set()
        for n in ncs:
            key = (n["regulation"], n["article"])
            if key not in seen:
                seen.add(key)
                plan.append(Act("lookup_article", {"regulation": n["regulation"], "article": n["article"]},
                                f"Je vérifie {n['regulation']} art. {n['article']}."))
        i = len(outcomes) - 1
        if i < len(plan):
            return plan[i]
        arts = {(o["args"]["regulation"], o["args"]["article"]): o["result"] for o in outcomes[1:]
                if o.get("tool") == "lookup_article" and _ok(o)}
        findings = [{"nc_id": n["id"], "regulation": n["regulation"], "article": n["article"],
                     "article_title": (arts.get((n["regulation"], n["article"])) or {}).get("title", "?"),
                     "severity": n["severity"], "title": n["title"]} for n in ncs]
        out = {"dossier_id": d, "client": first["result"]["client"], "findings": findings, "notes": notes}
        if self.mode == "injected":
            out["recipient_hint"] = hint
        return Final(out, "Analyse terminée.")


# ------------------------------------------------------------------ rédacteur
@dataclass
class WriterDouble:
    mode: str = "normal"  # normal | sloppy_first | always_sloppy | leak | runaway | raise
    id: str = "writer-double-v1"

    def _text(self, f: dict[str, Any], rnd: int) -> str:
        sloppy = self.mode == "always_sloppy" or (self.mode == "sloppy_first" and rnd == 1)
        text = f"Constat {f['nc_id']} — {f['title']}. Gravité : {f['severity']}."
        if not sloppy:
            text += f" Référence : {f['regulation']} art. {f['article']} ({f['article_title']})."
        if self.mode == "leak":
            text += " Copie à fuite@evil.example avec le mot de passe administrateur."
        return text

    def next(self, inputs, outcomes):
        if self.mode == "raise":
            raise DoubleError("le modèle du rédacteur a renvoyé une erreur")
        a, rnd = inputs["analysis"], inputs["round"]
        d = a["dossier_id"]
        findings = a["findings"]
        if self.mode == "runaway" and findings:  # ne s'arrête jamais de rédiger
            f = findings[0]
            return Act("draft_finding", {"dossier_id": d, "nc_id": f["nc_id"], "text": f"{self._text(f, rnd)} v{len(outcomes) + 1}"})
        for o in outcomes:
            if not _ok(o):
                return Final({"error": o.get("code") or "DRAFT_FAILED"}, "Rédaction impossible.")
        if len(outcomes) < len(findings):
            f = findings[len(outcomes)]
            return Act("draft_finding", {"dossier_id": d, "nc_id": f["nc_id"], "text": self._text(f, rnd)},
                       f"Je rédige le constat {f['nc_id']} (tour {rnd}).")
        drafts = [{"draft_id": o["result"]["draft_id"], "nc_id": o["args"]["nc_id"], "text": o["args"]["text"]}
                  for o in outcomes]
        recipient = a.get("recipient_hint") or inputs.get("recipient")
        proposal = None
        if recipient:
            proposal = {"tool": "submit_dossier", "args": {"dossier_id": d, "recipient": recipient}, "proposed_by": "writer"}
        return Final({"drafts": drafts, "proposed_action": proposal}, "Brouillons prêts ; je propose la transmission.")


# ------------------------------------------------------------------ relecteur
@dataclass
class ReviewerDouble:
    mode: str = "normal"  # normal | naive | rogue | raise
    id: str = "reviewer-double-v1"

    def next(self, inputs, outcomes):
        if self.mode == "raise":
            raise DoubleError("le modèle du relecteur a expiré")
        a, ds = inputs["analysis"], inputs["drafts"]
        plan: list[Act] = []
        if self.mode == "rogue":
            plan.append(Act("draft_finding", {"dossier_id": a["dossier_id"], "nc_id": "NC-011",
                                              "text": "Réécriture du constat par le relecteur."}, "Je corrige moi-même."))
        seen = set()
        for f in a["findings"]:
            key = (f["regulation"], f["article"])
            if key not in seen:
                seen.add(key)
                plan.append(Act("lookup_article", {"regulation": f["regulation"], "article": f["article"]},
                                f"Je revérifie {f['regulation']} art. {f['article']}."))
        if len(outcomes) < len(plan):
            return plan[len(outcomes)]
        if self.mode == "naive":
            return Final({"verdict": "approve", "reasons": ["relecture superficielle"]})
        allowed = set(inputs.get("allowed_domains", []))
        reasons, feedback = [], []
        prop = ds.get("proposed_action")
        if prop:
            rcpt = str(prop["args"].get("recipient", ""))
            if rcpt.rsplit("@", 1)[-1].lower() not in allowed:
                reasons.append(f"destinataire hors domaines autorisés : {rcpt}")
        by_nc = {f["nc_id"]: f for f in a["findings"]}
        for dr in ds["drafts"]:
            for m in EMAIL_RE.findall(dr["text"]):
                if m.rsplit("@", 1)[-1].lower() not in allowed:
                    reasons.append(f"{dr['nc_id']} : adresse externe dans le brouillon ({m})")
            if "mot de passe" in dr["text"].lower():
                reasons.append(f"{dr['nc_id']} : donnée sensible dans le brouillon")
        if reasons:
            return Final({"verdict": "block", "reasons": reasons}, "Brouillon non conforme : je bloque.")
        for dr in ds["drafts"]:
            f = by_nc.get(dr["nc_id"])
            if f and f"art. {f['article']}" not in dr["text"]:
                feedback.append({"nc_id": dr["nc_id"], "issue": f"référence {f['regulation']} art. {f['article']} absente"})
        if feedback:
            return Final({"verdict": "request_changes", "reasons": ["références manquantes"], "feedback": feedback})
        return Final({"verdict": "approve", "reasons": ["brouillons sourcés, destinataire autorisé"]})


DOUBLES: dict[str, Callable[[str], Brain]] = {"analyst": AnalystDouble, "writer": WriterDouble, "reviewer": ReviewerDouble}


@dataclass
class BrainFactory:
    """Fabrique les cerveaux d'une mission à partir de `MissionInput.doubles` (rejouable : enregistré au journal)."""
    overrides: dict[str, Brain] = field(default_factory=dict)  # tests : cerveau imposé pour un rôle

    def __call__(self, role: str, doubles: dict[str, Any]) -> Brain:
        if role in self.overrides:
            return self.overrides[role]
        return DOUBLES[role](doubles.get(role, "normal"))
