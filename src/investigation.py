"""Deterministic clarification and evidence-derived investigation views.

This is not live diagnosis. Causes remain historical hypotheses; generic checks
are read-only suggestions, never commands or tool calls.
"""

import re

from pydantic import BaseModel, Field


class Clarification(BaseModel):
    required_fields: list[str] = Field(default_factory=list)
    question: str = ""


class InvestigationFact(BaseModel):
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    status: str = "historical_evidence"


class Troubleshooting(BaseModel):
    symptoms: list[InvestigationFact] = Field(default_factory=list)
    relevant_incidents: list[InvestigationFact] = Field(default_factory=list)
    likely_causes: list[InvestigationFact] = Field(default_factory=list)
    suggested_read_only_checks: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


SCOPE_KEYS = ("element_id", "interface", "region", "software_version")
NODE_PATTERN = r"\b(?!(?:RB|RUNBOOK)[-_])(?:[A-Z]{1,12}[-_](?:[A-Z0-9]{1,12}[-_])*\d+|(?:CORE|NODE)[-_][A-Z0-9]+)\b"
OPERATIONAL = re.compile(r"\b(?:why|troubleshoot|diagnos\w*|investigat\w*|outage|failing|down|slow|packet loss|status|health)\b", re.I)
TICKET = re.compile(r"\b(?:INC|CHG|PRB)\d+\b", re.I)


def query_scope(query, context=None):
    scope = {key: value.strip() for key, value in (context or {}).items()
             if key in SCOPE_KEYS and isinstance(value, str) and 0 < len(value.strip()) <= 128}
    patterns = {
        "element_id": NODE_PATTERN,
        "interface": r"\b(?:eth\d+(?:/\d+)*|(?:ge|xe|et)-\d+(?:/\d+)+|(?:Gi|Te|GigabitEthernet)\d+(?:/\d+)+|Gx|Gy|Gr|S1-MME|N2|N3)\b",
        "region": r"\bregion[ _-]+(?!is\b|was\b|or\b|and\b|please\b|unknown\b)([A-Za-z0-9_-]+)",
        "software_version": r"\b(?:version|release|software)[ :]+(\d[\w.()-]*)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, query, re.I)
        if match:
            scope[key] = match.group(1) if key in ("region", "software_version") else match.group()
    return scope


def clarify(query, context=None, documents=None):
    scope = query_scope(query, context)
    conceptual = bool(re.search(r"^(?:explain|describe|how does|why does|why do)\b", query, re.I))
    operational = bool(OPERATIONAL.search(query)) and not TICKET.search(query) and not conceptual
    required = []
    if operational and len({match.group().casefold() for match in re.finditer(NODE_PATTERN, query, re.I)}) > 1:
        required.append("element_id")
    if operational and not any(scope.get(key) for key in ("element_id", "region")):
        if "element_id" not in required:
            required.append("element_id")
    if operational and re.search(r"\binterface\b", query, re.I) and not scope.get("interface"):
        required.append("interface")
    if operational and re.search(r"\bregion\b", query, re.I) and not scope.get("region"):
        required.append("region")
    if re.search(r"\b(?:version|release|software)\b", query, re.I) and operational and not scope.get("software_version"):
        required.append("software_version")
    if documents and operational and not scope.get("element_id"):
        nodes = {str(doc.metadata.get("element_id")) for doc in documents if doc.metadata.get("element_id")}
        if len(nodes) > 1 and "element_id" not in required:
            required.append("element_id")
    labels = {"element_id": "affected node ID", "interface": "interface name", "region": "region", "software_version": "software version"}
    return Clarification(required_fields=required,
        question="Please provide the " + ", ".join(labels[key] for key in required) + ". I cannot safely choose the operational scope." if required else "")


def scoped_documents(query, context, documents):
    """Do not answer about another asset merely because it ranked highly."""
    scope = query_scope(query, context)
    def matches(doc):
        for key, value in scope.items():
            recorded = doc.metadata.get(key)
            if key in ("element_id", "interface", "region") and recorded:
                if str(recorded).casefold().removeprefix("region ") != value.casefold().removeprefix("region "):
                    return False
            elif key in ("element_id", "interface") and not re.search(r"(?<![\w/-])" + re.escape(value) + r"(?![\w/-])", doc.content, re.I):
                return False
        return True
    return [doc for doc in documents if matches(doc)]


def retrieval_query(query, context):
    scope = query_scope(query, context)
    additions = [value for value in scope.values() if value.casefold() not in query.casefold()]
    return query + ("\nConfirmed scope: " + " ".join(additions) if additions else "")


def troubleshooting(query, evidence):
    view = Troubleshooting(symptoms=[InvestigationFact(text=query, status="user_report_unverified")])
    for item in evidence:
        if item.document_kind != "ticket":
            continue
        view.relevant_incidents.append(InvestigationFact(text=item.document_id, citation_ids=[item.citation_id]))
        match = re.search(r"(?:Root cause|root_cause):\s*([^\n]+)", item.passage, re.I)
        if match:
            view.likely_causes.append(InvestigationFact(text=match.group(1), citation_ids=[item.citation_id], status="historical_hypothesis_not_current_diagnosis"))
    view.suggested_read_only_checks = [
        "Inspect existing alarms and interface counters for the confirmed asset; do not reset counters.",
        "Compare existing event logs and monitoring timestamps with the reported symptom window.",
        "Review the recorded software version and recent change history without applying changes.",
    ]
    view.missing_evidence = ["Live alarms, counters and logs have not been collected.",
        "Confirm the symptom time window, exact asset/interface and software version.",
        "Historical causes have not been verified against this incident."]
    return view
