"""Build reproducible, versioned telecom ticket datasets and the golden set.

The source records are synthetic so they are safe to publish, but the fields,
failure modes, identifiers, and resolutions mirror normal NOC/ITSM data.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
GOLDEN_PATH = ROOT / "evaluation" / "golden_questions.jsonl"
CREATED_AT = "2026-08-05T00:00:00Z"
SOURCE_SNAPSHOT = "synthetic-servicenow-export-2026-07-31"


# id, severity, region, element, interface, service, summary, symptoms,
# root cause, resolution, runbook, question, reference answer, tags
SEEDS: list[tuple[Any, ...]] = [
    ("INC20260001", "P1", "London", "PCRF-LON-01", "Gx", "LTE policy control", "PCRF rejected new policy sessions", "DIAMETER_TOO_BUSY responses rose to 38% on the Gx peer", "A stale Diameter route kept traffic on a drained policy node", "Removed the stale route and reset the Gx peer; session success returned to 99.8%", "RB-PCC-014", "Which incident fixed the London policy node returning DIAMETER_TOO_BUSY on Gx, and how was it resolved?", "INC20260001 removed a stale Diameter route and reset the Gx peer, restoring session success to 99.8%.", ["entity_id", "acronym"]),
    ("INC20260002", "P1", "Manchester", "HLR-MAN-02", "Gr", "2G/3G subscriber lookup", "HLR lookup timeouts delayed location updates", "MAP UpdateLocation calls exceeded the three-second timeout", "A locked database index blocked the subscriber_location table", "Rebuilt the index online and restarted the blocked worker pool", "RB-HLR-006", "What ticket covers MAP UpdateLocation timeouts on HLR-MAN-02 and what removed the blockage?", "INC20260002 rebuilt the locked subscriber_location index and restarted the blocked workers.", ["entity_id", "acronym"]),
    ("INC20260003", "P2", "Birmingham", "SMSC-BHM-01", "SMPP", "Enterprise SMS", "Outbound SMS delivery queue grew continuously", "SMPP submit_sm latency exceeded 12 seconds and the queue reached 1.8 million messages", "One bind used an obsolete throughput cap after a partner migration", "Applied the current bind profile and drained the queue in controlled batches", "RB-SMS-021", "Which incident addressed the 1.8 million-message SMPP queue in Birmingham?", "INC20260003 applied the current SMPP bind profile and drained the outbound queue in batches.", ["entity_id", "acronym"]),
    ("INC20260004", "P1", "London", "MME-LON-02", "S1-MME", "LTE attach", "LTE attach failures spiked after a tracking-area change", "Attach Reject cause 15 appeared for TAC 4102", "The new tracking area was absent from the MME served-TA list", "Added TAC 4102 to the served-TA list and performed a rolling config reload", "RB-EPC-009", "Users on TAC 4102 saw Attach Reject cause 15. Which ticket corrected the MME?", "INC20260004 added TAC 4102 to MME-LON-02 and reloaded the configuration safely.", ["entity_id", "semantic"]),
    ("INC20260005", "P2", "Leeds", "SGW-LDS-03", "S5/S8", "LTE user plane", "Intermittent packet loss affected anchored LTE sessions", "GTP-U sequence gaps reached 4.2% during busy hour", "A line-card queue profile was incorrectly set to best effort", "Restored the mobile-core queue profile and cleared the congested queue", "RB-EPC-018", "Find the incident where GTP-U sequence gaps on SGW-LDS-03 reached 4.2%. What fixed it?", "INC20260005 restored the mobile-core queue profile and cleared the congested queue.", ["entity_id", "acronym"]),
    ("INC20260006", "P1", "Glasgow", "PGW-GLA-01", "Gi", "Mobile internet", "New mobile data sessions failed during evening peak", "Create Session succeeded but address allocation returned NAT_POOL_EXHAUSTED", "The Gi public NAT pool had no free addresses", "Added the reserved address block and rebalanced allocations across pools", "RB-PGW-012", "Which outage was caused by NAT_POOL_EXHAUSTED on the Glasgow Gi interface?", "INC20260006 expanded and rebalanced the Gi NAT pools on PGW-GLA-01.", ["entity_id", "acronym"]),
    ("INC20260007", "P1", "Bristol", "ENB-BRS-1042", "S1-U", "LTE radio access", "All sectors on one eNodeB went out of service", "Cells BRS1042A, BRS1042B, and BRS1042C reported VSWR alarms", "Water ingress damaged the feeder junction after overnight storms", "Replaced the junction, resealed the cabinet, and completed a sweep test", "RB-RAN-031", "Three Bristol sectors reported VSWR alarms after a storm. What ticket and repair restored them?", "INC20260007 replaced and resealed the damaged feeder junction, then passed a sweep test.", ["entity_id", "acronym"]),
    ("INC20260008", "P2", "London", "PCSCF-LON-04", "Mw", "IMS registration", "IMS registrations intermittently returned service unavailable", "SIP 503 responses appeared every five minutes on one P-CSCF", "A health probe recycled healthy workers because its DNS timeout was too short", "Raised the probe timeout and rolled the P-CSCF deployment", "RB-IMS-017", "Which incident explains five-minute SIP 503 bursts from PCSCF-LON-04?", "INC20260008 raised the health-probe DNS timeout and rolled the P-CSCF deployment.", ["entity_id", "acronym"]),
    ("INC20260009", "P2", "Edinburgh", "SBC-EDI-02", "SIP-I", "Voice interconnect", "Calls to one carrier timed out before answer", "SIP INVITE transactions ended with 408 on trunk TG-441", "The carrier changed its signalling IP without completing the change record", "Added the approved peer address and removed the obsolete route", "RB-VOICE-022", "What ticket fixed SIP 408 timeouts on Edinburgh trunk TG-441?", "INC20260009 added the carrier's approved peer address and removed the obsolete signalling route.", ["entity_id", "acronym"]),
    ("INC20260010", "P2", "Cardiff", "DNS-CDF-03", "UDP/53", "Mobile APN resolution", "Handsets could not resolve the enterprise APN", "Queries for corp5g.apn returned NXDOMAIN only in Wales", "A zone transfer excluded the delegated corp5g record", "Corrected the transfer ACL and forced a full zone transfer", "RB-DNS-004", "Which incident resolved Welsh NXDOMAIN responses for corp5g.apn?", "INC20260010 corrected the DNS transfer ACL and forced a full zone transfer.", ["entity_id", "semantic"]),
    ("INC20260011", "P1", "National Core", "CR-LON-01", "xe-0/0/7", "IP backbone", "Core routes flapped between London and Manchester", "BGP neighbour 172.20.14.2 reset 46 times in an hour", "A damaged optic caused intermittent loss of signal on xe-0/0/7", "Replaced the optic and cleaned both fibre ends", "RB-IP-011", "Which incident stopped 46 BGP resets on neighbour 172.20.14.2?", "INC20260011 replaced the damaged optic on CR-LON-01 xe-0/0/7 and cleaned the fibre.", ["entity_id", "acronym"]),
    ("INC20260012", "P2", "Manchester", "ER-MAN-07", "ae12", "Metro Ethernet", "OSPF adjacency remained in EXSTART after a router upgrade", "The adjacency on ae12 never reached FULL", "The two ends used MTU 9100 and MTU 9000", "Aligned both interfaces to MTU 9100 and reset the adjacency", "RB-IP-019", "An OSPF peer on ER-MAN-07 ae12 was stuck in EXSTART. What was the ticket and fix?", "INC20260012 aligned both ends of ae12 to MTU 9100 and reset OSPF.", ["entity_id", "acronym"]),
    ("INC20260013", "P1", "National Core", "P-RTR-BHM-02", "LDP", "MPLS backbone", "New label-switched paths could not be established", "LDP logged label allocation failures above 95% utilisation", "Orphaned labels remained after an aborted maintenance job", "Cleared the orphaned bindings and expanded the label range", "RB-MPLS-008", "Which incident dealt with LDP label allocation failures above 95 percent?", "INC20260013 cleared orphaned MPLS bindings and expanded the label range.", ["entity_id", "acronym"]),
    ("INC20260014", "P3", "Liverpool", "PE-LIV-05", "fxp0", "Enterprise VPN", "Provider-edge control-plane CPU stayed above 92%", "SNMP polling generated 18,000 requests per minute", "A monitoring template applied a one-second interval to every interface OID", "Restored the 60-second interval and staggered polling", "RB-NMS-013", "What ticket reduced 18,000 SNMP requests per minute against PE-LIV-05?", "INC20260014 restored 60-second staggered SNMP polling, reducing control-plane CPU.", ["entity_id", "acronym"]),
    ("INC20260015", "P1", "Newcastle", "ROADM-NCL-03", "CH-47", "DWDM transport", "Optical channel power deteriorated across the north path", "CH-47 receive power fell to -18.7 dBm", "A patch lead was sharply bent in the Newcastle rack", "Re-routed and replaced the patch lead, restoring receive power to -6.1 dBm", "RB-OPT-007", "Which ticket restored DWDM channel CH-47 from minus 18.7 dBm?", "INC20260015 replaced the bent patch lead and restored CH-47 receive power to -6.1 dBm.", ["entity_id", "acronym"]),
    ("INC20260016", "P2", "Cumbria", "MW-CMB-18", "RF-2", "Microwave backhaul", "Rural backhaul errors increased during heavy rain", "Adaptive modulation dropped from 256-QAM to QPSK", "A misaligned antenna reduced fade margin below design", "Realigned the antenna and verified a 28 dB fade margin", "RB-MW-010", "What incident fixed the microwave hop falling from 256-QAM to QPSK?", "INC20260016 realigned the antenna and verified 28 dB fade margin.", ["entity_id", "acronym"]),
    ("INC20260017", "P1", "Belfast", "OLT-BFS-09", "PON-3/2", "FTTP broadband", "Sixty-four premises lost fibre service", "All ONTs on PON-3/2 entered LOS simultaneously", "A splitter tray was disconnected during contractor work", "Reconnected and secured the tray, then verified all 64 ONTs online", "RB-FTTP-015", "Which ticket restored all 64 ONTs on OLT-BFS-09 PON-3/2?", "INC20260017 reconnected and secured the splitter tray, restoring all 64 ONTs.", ["entity_id", "acronym"]),
    ("INC20260018", "P2", "Sheffield", "BRAS-SHF-02", "RADIUS", "Consumer broadband", "Broadband customers looped at authentication", "PPPoE sessions received repeated Access-Reject messages", "The RADIUS shared secret differed after a BRAS replacement", "Synchronized the secret through the credential vault and cleared stale sessions", "RB-BRAS-003", "PPPoE users in Sheffield received RADIUS Access-Reject. Which incident corrected it?", "INC20260018 synchronized the RADIUS secret and cleared stale PPPoE sessions.", ["entity_id", "acronym"]),
    ("INC20260019", "P1", "London", "CGNAT-LON-06", "NAT44", "Mobile internet", "Customers could attach but could not open new internet connections", "Port-block allocation exceeded 99.5%", "A gaming promotion created unusually long UDP mappings", "Reduced the UDP idle timer under the approved policy and added a port block", "RB-CGN-005", "What ticket handled CGNAT port-block use above 99.5 percent?", "INC20260019 reduced the approved UDP idle timer and added CGNAT capacity.", ["entity_id", "acronym"]),
    ("INC20260020", "P1", "London", "FW-LON-DC2", "VLAN-318", "Diameter signalling", "Policy and charging traffic was silently dropped", "Gx requests matched the default deny rule on VLAN-318", "A firewall object group omitted the new PCRF subnet", "Added the approved subnet and validated Diameter connectivity", "RB-SEC-026", "Which incident fixed Gx traffic hitting default deny on FW-LON-DC2 VLAN-318?", "INC20260020 added the approved PCRF subnet to the firewall object group.", ["entity_id", "acronym"]),
    ("INC20260021", "P1", "Bristol", "MGW-BRS-01", "RTP", "VoLTE calls", "VoLTE calls had one-way audio toward fixed networks", "SIP signalling completed but inbound RTP counters remained zero", "A return-path ACL omitted the media gateway subnet", "Corrected the ACL and verified bidirectional RTP on test calls", "RB-VOLTE-019", "One-way VoLTE audio showed zero inbound RTP on MGW-BRS-01. What ticket fixed the return path?", "INC20260021 corrected the return-path ACL and verified bidirectional RTP.", ["entity_id", "acronym"]),
    ("INC20260022", "P2", "London", "EPDG-LON-02", "SWu", "Voice over Wi-Fi", "VoWiFi devices failed secure tunnel establishment", "IKE_AUTH failed with certificate unknown for iOS clients", "The ePDG certificate chain lacked the renewed intermediate CA", "Installed the full chain and reloaded the IPsec service", "RB-VOWIFI-008", "Which incident resolved IKE_AUTH certificate unknown on EPDG-LON-02?", "INC20260022 installed the renewed intermediate CA chain and reloaded IPsec.", ["entity_id", "acronym"]),
    ("INC20260023", "P1", "London", "AMF-LON-01", "N2", "5G registration", "5G registrations were rejected in a newly launched area", "Registration Reject cause tracking area not allowed affected TAC 5017", "TAC 5017 was missing from the AMF tracking-area configuration", "Added TAC 5017 and rolled the AMF instances one at a time", "RB-5GC-004", "Which ticket fixed 5G registration rejects for TAC 5017?", "INC20260023 added TAC 5017 and performed a rolling AMF restart.", ["entity_id", "acronym"]),
    ("INC20260024", "P1", "Manchester", "SMF-MAN-02", "N11", "5G enterprise data", "Enterprise devices could register but not establish data sessions", "PDU Session Reject reported missing or unknown DNN factory.private", "The DNN mapping was absent from the production SMF profile", "Added the approved DNN-to-UPF mapping and reloaded the profile", "RB-5GC-011", "PDU sessions for DNN factory.private were rejected. Which incident repaired the SMF mapping?", "INC20260024 added the approved factory.private DNN-to-UPF mapping.", ["entity_id", "acronym"]),
    ("INC20260025", "P2", "Leeds", "UPF-LDS-03", "N3", "5G user plane", "Large packets failed across the 5G user plane", "ICMP worked while payloads above 1420 bytes were lost", "The transport path did not support the UPF's configured MTU 1500", "Set the N3 MTU to 1400 and enabled path MTU discovery", "RB-5GC-016", "Which ticket addressed UPF-LDS-03 dropping payloads larger than 1420 bytes?", "INC20260025 set N3 MTU to 1400 and enabled path MTU discovery.", ["entity_id", "semantic"]),
    ("INC20260026", "P2", "Glasgow", "UDM-GLA-01", "Nudm", "5G subscriber data", "Recently upgraded subscribers retained old service profiles", "UDM responses served the previous AMBR for up to six hours", "Cache invalidation events were not consumed after a topic rename", "Corrected the topic name and invalidated affected subscriber cache entries", "RB-5GC-023", "What ticket fixed UDM serving a previous AMBR for six hours?", "INC20260026 corrected the cache-invalidation topic and purged affected UDM entries.", ["entity_id", "acronym"]),
    ("INC20260027", "P1", "Birmingham", "AUSF-BHM-01", "Nausf", "5G authentication", "Authentication vector checks failed intermittently", "AUTN validation errors peaked at 27% across two AUSF pods", "Node clocks drifted 94 seconds after NTP filtering changed", "Restored NTP access and restarted only the affected pods", "RB-5GC-027", "Which incident linked 27 percent AUTN failures to 94 seconds of clock drift?", "INC20260027 restored NTP access and restarted the affected AUSF pods.", ["entity_id", "acronym"]),
    ("INC20260028", "P2", "London", "NRF-LON-01", "Nnrf", "5G service discovery", "Network functions received intermittent discovery not found responses", "NRF returned HTTP 404 for healthy SMF instances", "Expired registrations persisted on one replica after database failover", "Removed stale registrations and resynchronized the replica", "RB-5GC-030", "What ticket removed stale NRF registrations that caused HTTP 404 for healthy SMFs?", "INC20260028 removed expired registrations and resynchronized the NRF replica.", ["entity_id", "acronym"]),
    ("INC20260029", "P2", "Manchester", "NSSF-MAN-01", "Nnssf", "5G network slicing", "Industrial devices were assigned to the consumer slice", "S-NSSAI 1-010203 resolved to the default UPF group", "A slice mapping lost its enterprise tracking-area condition", "Restored the condition and replayed failed selection requests", "RB-5GC-033", "Which incident corrected S-NSSAI 1-010203 selecting the consumer UPF group?", "INC20260029 restored the enterprise tracking-area condition in the slice mapping.", ["entity_id", "acronym"]),
    ("INC20260030", "P2", "London", "PCF-LON-03", "Npcf", "5G policy control", "Unlimited plans were incorrectly rate limited", "The PCF installed a 10 Mbps QoS rule for premium subscribers", "A policy package used the legacy product-code mapping", "Rolled back the package and deployed the corrected product mapping", "RB-5GC-037", "Premium 5G users received a 10 Mbps rule from PCF-LON-03. What ticket reversed it?", "INC20260030 rolled back the policy package and deployed the corrected product mapping.", ["entity_id", "semantic"]),
    ("INC20260031", "P1", "London", "OCS-LON-02", "Ro", "Prepaid charging", "Prepaid calls failed at authorization", "Diameter CCR-Initial requests timed out after two seconds", "A database connection pool was exhausted by abandoned balance queries", "Terminated abandoned queries and raised the pool after capacity review", "RB-CHG-012", "Which incident fixed two-second CCR-Initial timeouts on OCS-LON-02?", "INC20260031 cleared abandoned balance queries and increased the reviewed connection pool.", ["entity_id", "acronym"]),
    ("INC20260032", "P2", "Leeds", "MED-LDS-04", "SFTP", "CDR mediation", "Charging records accumulated before billing export", "The mediation backlog reached 22 million CDRs", "A full audit partition left insufficient working disk space", "Archived the audit partition and replayed CDR batches with reconciliation", "RB-BSS-009", "What ticket recovered a mediation backlog of 22 million CDRs?", "INC20260032 archived the audit partition and replayed reconciled CDR batches.", ["entity_id", "acronym"]),
    ("INC20260033", "P2", "National BSS", "TOPUP-API-02", "REST", "Prepaid top-up", "Some retried top-ups credited balances twice", "Duplicate credits shared one external transaction reference", "The retry path bypassed the idempotency-key lookup", "Enforced idempotency before posting and reversed confirmed duplicates", "RB-BSS-014", "Which ticket stopped duplicate prepaid credits caused by retry handling?", "INC20260033 enforced idempotency before balance posting and reversed confirmed duplicates.", ["entity_id", "semantic"]),
    ("INC20260034", "P3", "National BSS", "BILL-RULE-07", "BATCH", "Enterprise billing", "A contracted discount was absent from July invoices", "Accounts on plan ENT-FLEX-24 received the standard rate", "The discount rule's effective date used the wrong timezone", "Corrected the effective date and generated adjustment notes", "RB-BSS-020", "What incident corrected missing ENT-FLEX-24 discounts on July invoices?", "INC20260034 corrected the discount effective date and generated adjustment notes.", ["entity_id", "semantic"]),
    ("INC20260035", "P1", "National Voice", "MNP-RTG-01", "SIP", "Mobile number portability", "Calls to recently ported numbers went to the donor network", "Routing lookups returned a stale portability prefix for four ranges", "The MNP delta feed stopped after a certificate renewal", "Installed the renewed client chain and replayed all missed deltas", "RB-MNP-005", "Which incident replayed missed MNP deltas after a certificate renewal?", "INC20260035 installed the renewed client chain and replayed missed portability deltas.", ["entity_id", "acronym"]),
    ("INC20260036", "P1", "Roaming Edge", "DRA-LON-01", "S6a", "Inbound LTE roaming", "Inbound roamers could not attach from one partner", "Diameter requests failed with DIAMETER_UNABLE_TO_DELIVER", "The partner realm was missing from the DRA routing table", "Added the approved realm route and validated test IMSIs", "RB-ROAM-011", "What ticket fixed DIAMETER_UNABLE_TO_DELIVER for an inbound S6a partner?", "INC20260036 added the missing partner realm route on DRA-LON-01.", ["entity_id", "acronym"]),
    ("INC20260037", "P2", "International SMS", "SMS-IW-03", "SMPP", "SMS interconnect", "Delivery receipts were not returned to one partner", "Messages delivered but DLRs remained in ENROUTE state", "The partner expected receipted_message_id in hexadecimal", "Enabled the partner-specific hexadecimal conversion and replayed receipts", "RB-SMS-028", "Which incident fixed DLRs stuck ENROUTE for the SMS interconnect?", "INC20260037 enabled hexadecimal receipt-ID conversion and replayed the DLRs.", ["entity_id", "acronym"]),
    ("INC20260038", "P1", "Wales", "ECS-CDF-01", "SIP", "Emergency calling", "Emergency calls connected without dispatchable location", "The PIDF-LO body was absent for Wi-Fi originated calls", "A header normalization rule removed the location-routing key", "Excluded emergency traffic from the rule and verified location delivery", "RB-E911-002", "What ticket restored PIDF-LO location on Wi-Fi emergency calls?", "INC20260038 excluded emergency traffic from header normalization and verified location delivery.", ["entity_id", "acronym"]),
    ("INC20260039", "P2", "Secure Services", "LI-MED-02", "TLS", "Lawful intercept mediation", "Approved warrant records stopped transferring to the mediation gateway", "TLS negotiation failed immediately after certificate rotation", "The gateway trust store lacked the new issuing CA", "Added the issuing CA through the controlled change and replayed encrypted records", "RB-SEC-041", "Which incident repaired TLS after the LI mediation certificate rotation?", "INC20260039 added the new issuing CA and replayed the encrypted mediation records.", ["entity_id", "semantic"]),
    ("INC20260040", "P2", "National NOC", "NMS-EVT-01", "SNMP", "Network monitoring", "A single access-ring fault created 48,000 alarms", "Downstream alarms were not suppressed under the root LOS event", "A topology import omitted parent-child relationships", "Re-imported topology and re-ran alarm correlation", "RB-NMS-002", "What incident stopped a single LOS event producing 48,000 NMS alarms?", "INC20260040 re-imported topology and reran parent-child alarm correlation.", ["entity_id", "acronym"]),
    ("INC20260041", "P2", "ITSM Platform", "ITSM-ADAPTER-01", "OAuth2", "Incident automation", "Automated ticket creation started returning unauthorized", "POST requests to the ITSM API returned HTTP 401", "The client secret rotated but the adapter still used the previous vault version", "Pinned the adapter to the current vault version and restarted it", "RB-PLAT-006", "Which ticket fixed HTTP 401 from the ITSM adapter after secret rotation?", "INC20260041 selected the current vault secret version and restarted the adapter.", ["entity_id", "semantic"]),
    ("INC20260042", "P3", "Observability", "SLA-CALC-02", "SQL", "SLA reporting", "Monthly availability reports counted planned work as downtime", "The maintenance exclusion window shifted by one hour", "The calculator parsed UTC timestamps as Europe/London local time", "Normalized calculations to UTC and regenerated the monthly report", "RB-OBS-010", "What incident fixed the one-hour maintenance-window shift in SLA reports?", "INC20260042 normalized SLA calculations to UTC and regenerated the report.", ["entity_id", "semantic"]),
    ("INC20260043", "P2", "National Core", "CFG-AUDIT-01", "SSH", "Configuration compliance", "A compliant router fleet was flagged for weak ciphers", "The audit reported aes128-cbc despite its removal from devices", "The collector parsed cached startup configs instead of running configs", "Switched collection to running config and invalidated the cache", "RB-CFG-012", "Which ticket stopped CFG-AUDIT-01 reporting a removed SSH cipher?", "INC20260043 collected running configurations and invalidated the stale cache.", ["entity_id", "semantic"]),
    ("INC20260044", "P1", "GenAI Platform", "RAG-API-01", "HTTP", "Operations assistant", "The assistant API restarted under normal traffic", "Kubernetes pods were OOMKilled when concurrent prompts exceeded 30", "Prompt contexts had no token budget before model invocation", "Added context budgeting and raised the reviewed memory limit", "RB-AI-003", "What ticket addressed RAG-API-01 pods being OOMKilled above 30 concurrent prompts?", "INC20260044 added prompt-context budgeting and increased the reviewed pod memory limit.", ["entity_id", "semantic"]),
    ("INC20260045", "P2", "OSS Integration", "KAFKA-OSS-02", "topic:nms.events.v2", "Alarm ingestion", "Network alarms arrived more than twenty minutes late", "Consumer lag exceeded 3.4 million events on nms.events.v2", "A schema error sent one partition into a retry loop", "Quarantined the malformed event and resumed the partition from its committed offset", "RB-OSS-018", "Which incident recovered 3.4 million events of lag on nms.events.v2?", "INC20260045 quarantined the malformed event and resumed the Kafka partition.", ["entity_id", "semantic"]),
    ("INC20260046", "P2", "GenAI Platform", "REDIS-AI-01", "6379", "Conversation memory", "Active assistant sessions lost recent context", "Redis evicted session keys while memory was below the host limit", "The volatile-lru policy combined with a small container maxmemory caused eviction", "Raised the reviewed container limit and separated session keys from cache keys", "RB-AI-007", "What ticket stopped active conversation keys being evicted from REDIS-AI-01?", "INC20260046 separated session and cache keys and increased the reviewed Redis memory limit.", ["entity_id", "semantic"]),
    ("INC20260047", "P2", "GenAI Platform", "CHROMA-KB-01", "8100", "Knowledge retrieval", "The assistant could not find recently ingested runbooks", "The collection contained 7,412 of 9,806 expected chunks", "The ingestion worker acknowledged batches before Chroma persisted them", "Moved acknowledgement after persistence and replayed the missing batch manifest", "RB-AI-011", "Which incident restored the missing 2,394 chunks in CHROMA-KB-01?", "INC20260047 acknowledged only persisted batches and replayed the missing chunk manifest.", ["entity_id", "semantic"]),
    ("INC20260048", "P1", "GenAI Platform", "EMBED-WORKER-02", "OpenAI-compatible", "Document ingestion", "All new document embeddings failed validation", "The vector store expected 1,536 dimensions but the endpoint returned 3,072", "An embedding deployment changed without a versioned dataset configuration", "Pinned the model deployment and rebuilt the affected collection from its manifest", "RB-AI-015", "What ticket fixed the 3,072 versus 1,536 embedding dimension mismatch?", "INC20260048 pinned the embedding deployment and rebuilt the affected collection from its manifest.", ["entity_id", "semantic"]),
    ("INC20260049", "P1", "GenAI Platform", "GUARDRAIL-01", "REST", "Operations assistant", "A malicious ticket note attempted to override assistant instructions", "The note contained ignore previous instructions and requested secret output", "Ingested ticket text was treated as trusted prompt instructions", "Marked retrieved text as untrusted data and added injection-pattern rejection", "RB-AI-019", "Which incident hardened retrieval against an 'ignore previous instructions' ticket note?", "INC20260049 marked retrieved content as untrusted and added prompt-injection rejection.", ["entity_id", "semantic"]),
    ("INC20260050", "P1", "GenAI Platform", "PII-FILTER-01", "REST", "Operations assistant", "Generated answers exposed subscriber identifiers from ticket notes", "A response repeated an MSISDN and engineer email from retrieved evidence", "Output filtering ran before rather than after response generation", "Moved PII filtering after generation and added a regression test for source excerpts", "RB-AI-022", "What ticket prevented answers from repeating MSISDNs and engineer emails?", "INC20260050 moved PII filtering after generation and added source-excerpt regression coverage.", ["entity_id", "acronym"]),
]


# Deliberately terse lookup questions exercise the known weakness of semantic
# embeddings around opaque network identifiers and short telecom acronyms.
# The remaining questions are natural-language diagnostic questions from SEEDS.
GOLDEN_QUERY_OVERRIDES = {
    1: "Show the incident and remediation for PCRF-LON-01 on Gx.",
    2: "Find the resolved incident for HLR-MAN-02 on Gr.",
    3: "What was done for SMSC-BHM-01 on its SMPP link?",
    4: "Show the TAC 4102 incident on MME-LON-02.",
    5: "Find the packet-loss remediation for SGW-LDS-03 S5/S8.",
    6: "Which incident affected PGW-GLA-01 Gi?",
    7: "Give me the repair record for ENB-BRS-1042 S1-U.",
    8: "Show the five-minute failure on PCSCF-LON-04 Mw.",
    9: "Find the TG-441 incident on SBC-EDI-02.",
    10: "Which record covers DNS-CDF-03 UDP/53?",
    11: "Show the neighbour 172.20.14.2 incident on xe-0/0/7.",
    12: "Find the adjacency repair for ER-MAN-07 ae12.",
    13: "What happened to P-RTR-BHM-02 LDP?",
    14: "Show the CPU remediation for PE-LIV-05 fxp0.",
    15: "Find the optical repair for ROADM-NCL-03 CH-47.",
    16: "MW-CMB-18: 256-QAM to QPSK. Find the record.",
    17: "OLT-BFS-09 PON-3/2: ONT LOS. Find the record.",
    18: "BRAS-SHF-02: PPPoE RADIUS Access-Reject.",
    19: "CGNAT-LON-06 NAT44 port-block exhaustion.",
    20: "FW-LON-DC2 PCRF Gx VLAN-318 default deny.",
    21: "MGW-BRS-01 VoLTE SIP/RTP one-way audio.",
    22: "EPDG-LON-02 SWu IKE_AUTH certificate unknown.",
    23: "AMF-LON-01 5G TAC Registration Reject.",
    24: "SMF-MAN-02 PDU Session Reject DNN factory.private.",
    25: "UPF-LDS-03 N3 MTU 1420 packet loss.",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _records() -> list[dict[str, Any]]:
    records = []
    for index, seed in enumerate(SEEDS, 1):
        (
            ticket_id, severity, region, element_id, interface, service, summary,
            symptoms, root_cause, resolution, runbook, question, answer, tags,
        ) = seed
        records.append(
            {
                "ticket_id": ticket_id,
                "status": "Resolved",
                "severity": severity,
                "region": region,
                "element_id": element_id,
                "interface": interface,
                "service": service,
                "summary": summary,
                "description": f"{summary}. {symptoms}.",
                "symptoms": symptoms,
                "root_cause": root_cause,
                "resolution": resolution,
                "runbook_id": runbook,
                "opened_at": f"2026-07-{((index - 1) % 28) + 1:02d}T{8 + index % 10:02d}:15:00Z",
                "resolved_at": f"2026-07-{((index - 1) % 28) + 1:02d}T{10 + index % 10:02d}:45:00Z",
                "source": f"servicenow/incidents/{ticket_id}.json",
                "reporter_email": f"engineer{index}@noc.example.test",
                "contact_phone": f"+44770090{index:04d}",
                "management_ip": f"10.42.{index // 255}.{index % 255}",
                "_golden_question": question,
                "_reference_answer": answer,
                "_tags": tags,
            }
        )
    return records


def _clean(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(record)
    for key, value in cleaned.items():
        if isinstance(value, str):
            cleaned[key] = re.sub(r"\s+", " ", value).strip()
    cleaned["status"] = cleaned["status"].lower()
    cleaned["severity"] = cleaned["severity"].upper()
    return cleaned


def _redact(record: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(record)
    redacted["reporter_email"] = "[EMAIL_REDACTED]"
    redacted["contact_phone"] = "[PHONE_REDACTED]"
    redacted["management_ip"] = "[IP_REDACTED]"
    return redacted


def _publishable(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _ticket_aware_text(record: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Ticket: {record['ticket_id']} | Status: {record['status']} | Severity: {record['severity']}",
            f"Network: {record['region']} | Element: {record['element_id']} | Interface: {record['interface']}",
            f"Service: {record['service']} | Runbook: {record['runbook_id']}",
            f"Incident: {record['summary']}",
            f"Symptoms: {record['symptoms']}",
            f"Root cause: {record['root_cause']}",
            f"Resolution: {record['resolution']}",
        ]
    )


def _manifest(
    *,
    version: str,
    parent: str | None,
    snapshot_path: Path,
    cleaning_rules: list[str],
    chunking: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_version": version,
        "source_snapshot": SOURCE_SNAPSHOT,
        "parent_version": parent,
        "snapshot_file": str(snapshot_path.relative_to(ROOT)).replace("\\", "/"),
        "snapshot_sha256": _sha256(snapshot_path),
        "cleaning_rules": cleaning_rules,
        "chunking_configuration": chunking,
        "embedding_model": "telecom-hashing-embedding-v1",
        "embedding_dimensions": 384,
        "number_of_documents": len(SEEDS),
        "number_of_rejected_records": 0,
        "creation_timestamp": CREATED_AT,
        "synthetic_data": True,
        "evaluation_results": {
            "status": "not_evaluated",
            "report": "reports/evaluation/baseline.json",
        },
    }


def build() -> None:
    raw = _records()
    cleaned = [_clean(row) for row in raw]
    redacted = [_redact(row) for row in cleaned]

    versions: list[tuple[str, list[dict[str, Any]], list[str], dict[str, Any], str | None]] = [
        (
            "tickets-v1.0",
            raw,
            ["none; immutable synthetic source export"],
            {"strategy": "recursive", "chunk_size": 240, "chunk_overlap": 40},
            None,
        ),
        (
            "tickets-v1.1-cleaned",
            cleaned,
            ["collapse whitespace", "lowercase status", "uppercase severity"],
            {"strategy": "recursive", "chunk_size": 240, "chunk_overlap": 40},
            "tickets-v1.0",
        ),
        (
            "tickets-v1.2-pii-redacted",
            redacted,
            [
                "inherit tickets-v1.1-cleaned rules",
                "redact reporter email",
                "redact contact phone",
                "redact management IP",
            ],
            {"strategy": "recursive", "chunk_size": 240, "chunk_overlap": 40},
            "tickets-v1.1-cleaned",
        ),
        (
            "tickets-v2.0-ticket-aware-chunking",
            redacted,
            ["inherit tickets-v1.2-pii-redacted rules"],
            {
                "strategy": "ticket-aware",
                "identity_fields": ["ticket_id", "severity", "region", "element_id", "interface", "service", "runbook_id"],
                "incident_fields": ["summary", "description", "symptoms"],
                "resolution_fields": ["root_cause", "resolution"],
            },
            "tickets-v1.2-pii-redacted",
        ),
    ]

    for version, rows, rules, chunking, parent in versions:
        target_dir = DATASETS / version
        snapshot = target_dir / "tickets.jsonl"
        publishable = [_publishable(row) for row in rows]
        if version == "tickets-v2.0-ticket-aware-chunking":
            publishable = [dict(row, ticket_aware_text=_ticket_aware_text(row)) for row in publishable]
        _write_jsonl(snapshot, publishable)
        _write_json(
            target_dir / "manifest.json",
            _manifest(
                version=version,
                parent=parent,
                snapshot_path=snapshot,
                cleaning_rules=rules,
                chunking=chunking,
            ),
        )

    golden_rows = []
    for index, row in enumerate(redacted, 1):
        if index <= 15:
            tags = ["entity_id"]
        elif index <= 25:
            tags = ["acronym"]
        else:
            tags = ["semantic"]
        golden_rows.append(
            {
                "question_id": f"GQ-{index:03d}",
                "question": GOLDEN_QUERY_OVERRIDES.get(index, row["_golden_question"]),
                "expected_ticket_ids": [row["ticket_id"]],
                "expected_sources": [row["source"]],
                "reference_answer": row["_reference_answer"],
                "tags": tags,
            }
        )
    _write_jsonl(GOLDEN_PATH, golden_rows)

    print(f"Built {len(versions)} dataset versions and {len(golden_rows)} golden questions")


if __name__ == "__main__":
    build()
