"""
Enterprise Knowledge Base Corpus and Evaluation Query Generator
"""
from typing import List, Tuple
from rag_bench.models import Document, Query

ENTERPRISE_DOCS = [
    ("doc_it_01", "IT VPN & Remote Access Policy: Employees must connect using WireGuard or IPsec VPN with hardware MFA tokens. Split tunneling is strictly prohibited on production subnets."),
    ("doc_it_02", "Hardware Asset Management: Laptops must have BitLocker or FileVault full-disk encryption enabled and checked into MDM before connecting to internal networks."),
    ("doc_it_03", "SSH Key Management and Rotation: SSH private keys must be protected by hardware security keys (FIDO2/YubiKey). Host keys rotate biannually."),
    ("doc_iam_01", "AWS IAM Role Assumption: Cross-account role assumption requires ExternalId and MFA session validation. Root accounts must have zero active access keys."),
    ("doc_iam_02", "Least Privilege Access Review: IAM permissions are reviewed quarterly. Unused permissions older than 90 days are automatically revoked via IAM Access Analyzer."),
    ("doc_iam_03", "Privileged Identity Management: Emergency administrative access requires temporary approval via PIM with a maximum session duration of 2 hours."),
    ("doc_k8s_01", "Kubernetes RBAC Standards: Cluster-admin roles must not be assigned to default service accounts. All deployments must enforce read-only root filesystems."),
    ("doc_k8s_02", "Pod Security Standards: Pods running with root UID or privileged flags are blocked by admission controllers. Seccomp profiles must be set to RuntimeDefault."),
    ("doc_k8s_03", "Kubernetes Network Policies: Ingress and egress traffic between namespaces is blocked by default and explicitly allowed only via Calico network policies."),
    ("doc_db_01", "PostgreSQL Database Connection Pooling: All microservices must connect through PgBouncer connection poolers using TLS 1.3 certificates and mTLS mutual authentication."),
    ("doc_db_02", "Database Backup and Recovery: Nightly encrypted database snapshots are transferred to air-gapped S3 buckets with object lock retention enabled."),
    ("doc_db_03", "Sensitive Data Masking: Customer social security numbers and credit card details must be masked using column-level encryption in PostgreSQL databases."),
    ("doc_fin_01", "Vendor Invoice Payment Routing: Vendor invoice disbursements exceeding $10,000 require dual executive sign-off and verification with official accounting records."),
    ("doc_fin_02", "Employee Expense Reimbursement Guidelines: Travel and entertainment expenses must be submitted within 30 days accompanied by itemized receipts through Concur."),
    ("doc_fin_03", "Wire Transfer Protocols: Corporate wire transfers require out-of-band telephone verification with the verified counterparty bank representative."),
    ("doc_hr_01", "Employee Onboarding Workflow: New hires must complete mandatory security awareness training and background checks within 7 days of start date."),
    ("doc_hr_02", "Remote Work Stipend: Full-time employees are eligible for an annual $1,000 home office equipment stipend reimbursed through payroll."),
    ("doc_hr_03", "Performance Review Cycle: Bi-annual employee appraisals are conducted in June and December using standardized competency rubrics."),
    ("doc_ir_01", "Incident Response Severity Matrix: P1 critical incidents require incident commander assignment within 15 minutes and automated bridge creation on Slack."),
    ("doc_ir_02", "Forensic Evidence Handling: Chain of custody forms must accompany memory dumps and disk images collected from compromised systems."),
    ("doc_ir_03", "Post-Incident Review Procedures: Blameless post-mortems must be published within 5 business days detailing root causes and preventive action items."),
    ("doc_api_01", "API Authentication Standards: All public REST APIs must enforce OAuth 2.0 Bearer tokens with JWT signatures validated using JSON Web Key Sets (JWKS)."),
    ("doc_api_02", "API Rate Limiting: External endpoints enforce a rate limit of 100 requests per minute per IP address, returning HTTP 429 upon exhaustion."),
    ("doc_api_03", "Mutual TLS for Service-to-Service: Internal microservice communication requires mTLS with SPIFFE/SPIRE workload identities."),
    ("doc_sec_01", "CI/CD Pipeline Security: GitHub Actions workflows must use pinned SHA commitments for third-party actions and ephemeral self-hosted runners."),
    ("doc_sec_02", "Container Image Signing: All Docker images deployed to production must be signed with Cosign and verified via Sigstore admission policies."),
    ("doc_sec_03", "Static Application Security Testing: Pull requests failing SonarQube quality gates or introducing high-severity CVEs are blocked from merging."),
    ("doc_comp_01", "SOC 2 Type II Security Controls: Access reviews, change management logs, and vulnerability remediation SLAs are audited annually by external CPAs."),
    ("doc_comp_02", "GDPR Data Subject Request Processing: Right-to-be-forgotten requests must be completed across all production and backup data stores within 30 days."),
    ("doc_comp_03", "ISO 27001 Information Security Management: The Information Security Management System (ISMS) scope covers all cloud infrastructure and office facilities."),
]

SAMPLE_QUERIES = [
    ("q_01", "What are the rules for connecting to corporate network over VPN?", "doc_it_01", "it_policy"),
    ("q_02", "How should AWS IAM roles be assumed across accounts?", "doc_iam_01", "cloud_iam"),
    ("q_03", "What admission controls apply to privileged Kubernetes pods?", "doc_k8s_02", "kubernetes"),
    ("q_04", "How is mutual TLS configured for database connections?", "doc_db_01", "database"),
    ("q_05", "What is the authorization policy for vendor wire transfers?", "doc_fin_01", "finance"),
    ("q_06", "What is the response SLA for P1 critical security incidents?", "doc_ir_01", "incident_response"),
    ("q_07", "How are public API endpoints protected against brute force and abuse?", "doc_api_02", "api_governance"),
    ("q_08", "What security gates exist for CI/CD container image signing?", "doc_sec_02", "devsecops"),
    ("q_09", "How long do we have to fulfill GDPR data deletion requests?", "doc_comp_02", "compliance"),
    ("q_10", "What authentication is required for internal microservice communication?", "doc_api_03", "api_governance"),
]


def load_benchmark_dataset() -> Tuple[List[Document], List[Query]]:
    docs = [Document(doc_id=d_id, text=txt, metadata={"domain": d_id.split("_")[1]}) for d_id, txt in ENTERPRISE_DOCS]
    queries = [Query(query_id=q_id, text=q_txt, ground_truth_doc_id=gt_id, target_topic=topic) for q_id, q_txt, gt_id, topic in SAMPLE_QUERIES]
    return docs, queries
