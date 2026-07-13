"""Sample corpus shared by every demo in this package.

A small fictional internal knowledge base for "Aurora Cloud," a made-up SaaS
company, gives every retrieval variant the same coherent material to search:
a deployment policy, an incident runbook, an on-call rotation, a data
retention policy, an API rate-limit reference, and a billing FAQ. Keeping
one corpus across modules means a reader can compare how naive, hybrid, and
reranked retrieval each handle the same documents and queries.
"""

from __future__ import annotations

from patterns.rag.chunking import Chunk, Document, chunk_corpus

DEPLOY_POLICY = Document(
    id="deploy-policy",
    text=(
        "Aurora Cloud deployment policy. Every release ships through a canary "
        "rollout to five percent of traffic before a full rollout. If the "
        "canary shows elevated error rates, the on-call engineer runs aurora "
        "rollback release-id to revert to the previous stable release within "
        "two minutes. Deployment freezes apply during the last three days of "
        "each fiscal quarter and during any active SEV1 incident."
    ),
)

INCIDENT_RUNBOOK = Document(
    id="incident-runbook",
    text=(
        "Aurora Cloud incident response runbook. A SEV1 incident is any "
        "outage that blocks customer logins or payments. The on-call "
        "engineer declares the severity level within five minutes of the "
        "first alert and posts an update in the incident channel every "
        "fifteen minutes. The first mitigation step for a SEV1 caused by a "
        "recent deploy is always a rollback, not a forward fix. A postmortem "
        "is due within forty eight hours of resolution."
    ),
)

ONCALL_ROTATION = Document(
    id="oncall-rotation",
    text=(
        "Aurora Cloud on-call rotation policy. Primary and secondary "
        "on-call engineers rotate weekly on the aurora-primary PagerDuty "
        "schedule. If the primary does not acknowledge a page within "
        "fifteen minutes, PagerDuty escalates automatically to the "
        "secondary and then to the engineering manager. Trading a shift "
        "requires two weeks notice in the on-call calendar."
    ),
)

DATA_RETENTION = Document(
    id="data-retention",
    text=(
        "Aurora Cloud data retention policy. Application logs are retained "
        "for thirty days and then deleted automatically. Database backups "
        "are retained for ninety days on encrypted storage. A GDPR deletion "
        "request from a customer must be completed within thirty days of "
        "the request and confirmed to the customer in writing."
    ),
)

API_RATE_LIMITS = Document(
    id="api-rate-limits",
    text=(
        "Aurora Cloud API rate limits. The default limit is one hundred "
        "requests per minute per API key, with a short burst allowance of "
        "two hundred requests. A request over the limit receives a 429 "
        "response with a Retry-After header and the error code "
        "ERR_RATE_LIMIT_1004. Customers on the enterprise plan can request "
        "a higher limit through their account manager."
    ),
)

BILLING_FAQ = Document(
    id="billing-faq",
    text=(
        "Aurora Cloud billing FAQ. Aurora offers monthly and annual plans "
        "billed in advance. It also credits the unused portion of a plan "
        "automatically when a customer upgrades mid cycle. Refund requests "
        "are honored within fourteen days of the original charge. Invoice "
        "disputes go to the billing team through the support portal."
    ),
)

DOCUMENTS: list[Document] = [
    DEPLOY_POLICY,
    INCIDENT_RUNBOOK,
    ONCALL_ROTATION,
    DATA_RETENTION,
    API_RATE_LIMITS,
    BILLING_FAQ,
]

DOCUMENTS_BY_ID: dict[str, Document] = {doc.id: doc for doc in DOCUMENTS}

DEFAULT_CHUNK_SIZE = 220
DEFAULT_CHUNK_OVERLAP = 40


def default_chunks() -> list[Chunk]:
    """Chunk the sample corpus with this package's default size and overlap."""
    return chunk_corpus(DOCUMENTS, size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
