import logging
from typing import List, Dict, Any, Optional, Tuple
from backend.schemas import (
    CheckResult,
    CheckStatus,
    Severity,
    ComplianceReport,
    ProductCategory,
    AuthenticityDetails,
)
from backend.ml_layer.fuzzy_normalizer import FuzzyNormalizer
from backend.ml_layer.confidence_scorer import ComplianceConfidenceScorer

logger = logging.getLogger(__name__)

class VerificationReconciler:
    """
    Combines Layer 1 (deterministic rules) and Layer 2 (ML model)
    into a reconciled final compliance verdict.

    Reconciliation Principles:
    1. L1 PASS + L2 agrees (score >= 0.50) -> PASS (high confidence)
    2. L1 FAIL + L2 agrees (score < 0.50)  -> FAIL (high confidence)
    3. L1 UNCERTAIN + L2 recovers a value passing L1 -> PASS (recovered_by_ml=True)
    4. L1 UNCERTAIN + L2 cannot recover             -> REVIEW
    5. L1 & L2 disagree                             -> REVIEW (both opinions retained)
    6. HARD CONSTRAINT: ML layer can NEVER overturn a blocking Layer 1 FAIL into PASS.
       It can only downgrade it to REVIEW with an explanation.
    7. AUTH_01 stays independent: can force overall report verdict to FAIL without stopping checks.
    """

    def __init__(
        self,
        fuzzy_normalizer: Optional[FuzzyNormalizer] = None,
        confidence_scorer: Optional[ComplianceConfidenceScorer] = None,
    ):
        self.normalizer = fuzzy_normalizer or FuzzyNormalizer()
        self.scorer = confidence_scorer or ComplianceConfidenceScorer()

    def reconcile_check(
        self,
        check: CheckResult,
        raw_text: str = "",
        all_tokens_text: str = "",
    ) -> CheckResult:
        """
        Reconciles a single Layer 1 CheckResult with Layer 2 predictions.
        """
        # AUTH_01 is evaluated independently
        if check.check_id.upper() in ["AUTH_01", "AUTH01"] or check.check_name == "Image Authenticity":
            check.l1_status = check.status.value
            check.l2_score = 0.05 if check.status == CheckStatus.FAIL else 0.95
            check.decision_path = f"AUTH independent forensic arbiter -> {check.status.value}"
            return check

        l1_status = check.status
        check.l1_status = l1_status.value

        # Calculate Layer 2 compliance score
        check_dict = {
            "l1_status": l1_status.value,
            "evidence_text": check.evidence_text or "",
            "confidence": check.confidence or 0.0,
            "evidence_match_score": 0.90 if check.evidence_text else 0.10,
            "has_statutory_phrase": bool(
                check.evidence_text and any(w in check.evidence_text.lower() for w in ["tax", "incl", "ml", "g", "kg", "mfg", "india"])
            ),
        }
        l2_score = self.scorer.predict_compliance_score(check_dict)
        check.l2_score = l2_score

        source_text = f"{check.evidence_text or ''} {raw_text} {all_tokens_text}".strip()

        # ── Case A: L1 is UNCERTAIN ──
        if l1_status == CheckStatus.UNCERTAIN:
            recovered = self._attempt_ml_recovery(check.check_id, check.check_name, source_text)
            if recovered and recovered.get("is_valid"):
                # Value recovered and satisfies L1 standards!
                check.status = CheckStatus.PASS
                check.recovered_by_ml = True
                check.evidence_text = recovered.get("recovered_text", check.evidence_text)
                check.reason = f"Recovered by ML from noisy OCR: {check.evidence_text}"
                check.decision_path = "L1 uncertain -> L2 recovered -> L1 revalidated -> PASS"
                check.confidence = max(0.85, l2_score)
                return check
            else:
                # Cannot recover reliably -> REVIEW
                check.status = CheckStatus.REVIEW
                check.reason = f"{check.reason or 'Uncertain OCR match'} — ML recovery could not unambiguously resolve field."
                check.decision_path = "L1 uncertain -> L2 unrecovered -> REVIEW"
                check.confidence = l2_score
                return check

        # ── Case B: L1 is PASS ──
        elif l1_status == CheckStatus.PASS:
            if l2_score >= 0.45:
                # Agreement
                check.status = CheckStatus.PASS
                check.decision_path = f"L1 PASS + L2 agree ({l2_score:.2f}) -> PASS"
                check.confidence = max(check.confidence or 0.85, l2_score)
                return check
            else:
                # Disagreement: L1 said PASS, but L2 model strongly doubts it (score < 0.45)
                check.status = CheckStatus.REVIEW
                check.reason = f"{check.reason} [Reconciler Notice: L2 model confidence low ({l2_score:.2f}); manual review recommended]"
                check.decision_path = f"L1 PASS vs L2 non-compliant ({l2_score:.2f}) -> REVIEW"
                check.confidence = l2_score
                return check

        # ── Case C: L1 is FAIL ──
        elif l1_status == CheckStatus.FAIL:
            if l2_score < 0.55:
                # Agreement on failure
                check.status = CheckStatus.FAIL
                check.decision_path = f"L1 FAIL + L2 agree ({l2_score:.2f}) -> FAIL"
                check.confidence = round(1.0 - l2_score, 3)
                return check
            else:
                # Disagreement: L1 failed, but L2 suggests compliant (score >= 0.55)
                # Apply HARD CONSTRAINT:
                # ML layer can NEVER overturn a blocking Layer 1 FAIL into PASS.
                # It can only downgrade it to REVIEW with an explanation.
                if check.severity == Severity.BLOCKING:
                    check.status = CheckStatus.REVIEW
                    check.reason = (
                        f"{check.reason} [Hard Constraint: L2 model indicates probable compliance ({l2_score:.2f}); "
                        f"downgraded from blocking FAIL to REVIEW for human auditor confirmation]"
                    )
                    check.decision_path = f"L1 blocking FAIL vs L2 high ({l2_score:.2f}) -> Downgraded to REVIEW (Hard Constraint)"
                    check.confidence = l2_score
                    return check
                else:
                    # Non-blocking check disagreement -> REVIEW
                    check.status = CheckStatus.REVIEW
                    check.reason = f"{check.reason} [Reconciler Notice: L2 model predicts {l2_score:.2f} compliance; flagged for REVIEW]"
                    check.decision_path = f"L1 FAIL vs L2 high ({l2_score:.2f}) -> REVIEW"
                    check.confidence = l2_score
                    return check

        # ── Default / Passthrough (REVIEW, EXEMPT, etc.) ──
        check.decision_path = f"L1 {check.status.value} preserved"
        return check

    def _attempt_ml_recovery(self, check_id: str, check_name: str, text: str) -> Optional[Dict[str, Any]]:
        """
        Invokes FuzzyNormalizer specialized recovery based on check type.
        """
        c_name = check_name.lower()
        if "mrp" in c_name or check_id == "01":
            return self.normalizer.recover_mrp(text)
        elif "quantity" in c_name or "net" in c_name or check_id == "02":
            return self.normalizer.recover_net_quantity(text)
        elif "date" in c_name or check_id == "03":
            return self.normalizer.recover_date(text)
        elif "country" in c_name or check_id == "04":
            return self.normalizer.recover_country(text)
        elif "care" in c_name or "consumer" in c_name or check_id == "06":
            return self.normalizer.recover_consumer_care(text)
        return None

    def reconcile_report(
        self,
        checks: List[CheckResult],
        all_text: str = "",
        auth_details: Optional[AuthenticityDetails] = None,
    ) -> Tuple[List[CheckResult], CheckStatus]:
        """
        Reconciles all check results and computes the final overall report verdict.
        """
        reconciled_checks: List[CheckResult] = []
        for chk in checks:
            reconciled = self.reconcile_check(chk, raw_text=all_text, all_tokens_text=all_text)
            reconciled_checks.append(reconciled)

        # Compute overall verdict from reconciled checks
        overall = CheckStatus.PASS
        for r in reconciled_checks:
            if r.status == CheckStatus.FAIL and r.severity == Severity.BLOCKING:
                overall = CheckStatus.FAIL
                break
            elif r.status in [CheckStatus.REVIEW, CheckStatus.UNCERTAIN]:
                if overall != CheckStatus.FAIL:
                    overall = CheckStatus.REVIEW

        # Authenticity directive: AI-generated media forces overall FAIL
        if auth_details and auth_details.is_ai_generated:
            overall = CheckStatus.FAIL
        elif auth_details and auth_details.status == CheckStatus.REVIEW:
            if overall != CheckStatus.FAIL:
                overall = CheckStatus.REVIEW

        return reconciled_checks, overall
