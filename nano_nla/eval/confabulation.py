"""Confabulation analysis for NLA explanations.

NLA explanations sometimes make verifiably false claims about the context.
This module provides tools to:
  1. Extract individual claims from an explanation
  2. Verify claims against the original context
  3. Measure whether true claims recur across token positions
  4. Test if removing claims affects reconstruction MSE

Reference: Section "Characterizing NLA Confabulations"

Qwen-0.5B adaptation: Since we can't use a separate judge model (like
Haiku 4.5), we use string-matching heuristics and the AR's own
reconstruction sensitivity for claim verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from nano_nla.eval.evaluations import normalized_mse
from nano_nla.inference import NLAClient


@dataclass
class Claim:
    """An individual claim extracted from an NLA explanation."""
    text: str
    specificity: str       # "thematic", "entity", or "detail"
    is_supported: bool | None = None
    recurrence_count: int = 0
    mse_delta_on_removal: float | None = None


@dataclass
class ConfabulationReport:
    """Aggregate confabulation statistics for a set of explanations."""
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    unknown_claims: int
    support_rate: float
    thematic_support_rate: float
    entity_support_rate: float
    detail_support_rate: float
    recurring_claim_support_rate: float
    avg_mse_delta_true: float
    avg_mse_delta_false: float
    claims: list[Claim] = field(default_factory=list)


def extract_claims(explanation: str) -> list[Claim]:
    """Split an explanation into individual claims with specificity labels.

    Heuristic classification:
      - thematic: short phrases about topic/genre/domain
      - entity: mentions of specific names, places, dates
      - detail: quotes, specific numbers, claimed attributes
    """
    lines = [line.strip() for line in explanation.strip().split("\n") if line.strip()]
    # Remove bullet markers
    cleaned = []
    for line in lines:
        line = re.sub(r'^[\s\-*•–—]+', '', line).strip()
        if line:
            cleaned.append(line)

    # Split further on sentences within each line
    claims: list[Claim] = []
    for line in cleaned:
        sentences = re.split(r'(?<=[.!?])\s+', line)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # Classify specificity
            has_quote = '"' in sentence or "'" in sentence
            has_number = bool(re.search(r'\b\d{2,}\b', sentence))
            has_proper_noun = bool(re.search(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b', sentence))

            if has_quote or has_number:
                specificity = "detail"
            elif has_proper_noun:
                specificity = "entity"
            else:
                specificity = "thematic"

            claims.append(Claim(text=sentence, specificity=specificity))

    return claims


def verify_claim_against_context(
    claim: Claim,
    context: str,
    *,
    thematic_threshold: float = 0.3,
) -> bool | None:
    """Check if a claim is supported by the original context text.

    Uses word-overlap heuristics since we don't have a dedicated judge model.
    Returns True (supported), False (unsupported), or None (can't determine).
    """
    if not context or not claim.text:
        return None

    claim_words = set(re.findall(r'\b\w{3,}\b', claim.text.lower()))
    context_words = set(re.findall(r'\b\w{3,}\b', context.lower()))

    if not claim_words:
        return None

    overlap = len(claim_words & context_words) / len(claim_words)

    if claim.specificity == "thematic":
        return overlap >= thematic_threshold
    elif claim.specificity == "entity":
        # For entity claims, check if the proper nouns appear in context
        entities = re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b', claim.text)
        if not entities:
            return overlap >= 0.4
        return any(entity.lower() in context.lower() for entity in entities)
    else:  # detail
        return overlap >= 0.5


def count_claim_recurrence(
    claim: Claim,
    explanations_at_other_tokens: list[str],
    *,
    min_word_overlap: float = 0.5,
) -> int:
    """Count how often a claim appears in explanations at other token positions.

    The paper finds that recurring claims are more likely to be true.
    """
    claim_words = set(re.findall(r'\b\w{3,}\b', claim.text.lower()))
    if not claim_words:
        return 0

    count = 0
    for explanation in explanations_at_other_tokens:
        expl_words = set(re.findall(r'\b\w{3,}\b', explanation.lower()))
        if not expl_words:
            continue
        overlap = len(claim_words & expl_words) / len(claim_words)
        if overlap >= min_word_overlap:
            count += 1
    return count


@torch.no_grad()
def measure_claim_removal_impact(
    client: NLAClient,
    activation: torch.Tensor,
    explanation: str,
    claim: Claim,
) -> float:
    """Measure MSE change when a claim is removed from the explanation.

    The paper finds: removing true claims hurts MSE more than removing false ones.
    Returns delta_mse: positive means removal increased error.
    """
    if client.ar is None:
        raise RuntimeError("AR required for claim removal impact")

    gold = activation.float().cpu().view(1, -1)

    # Original reconstruction
    orig_recon = client.reconstruct(explanation)
    orig_mse = normalized_mse(orig_recon.view(1, -1), gold, client.nla_cfg.mse_scale).item()

    # Reconstruction with claim removed
    ablated = explanation.replace(claim.text, "").strip()
    ablated = re.sub(r'\n{3,}', '\n\n', ablated)  # clean up whitespace
    if not ablated:
        return 0.0
    ablated_recon = client.reconstruct(ablated)
    ablated_mse = normalized_mse(ablated_recon.view(1, -1), gold, client.nla_cfg.mse_scale).item()

    return ablated_mse - orig_mse


@torch.no_grad()
def analyze_confabulations(
    client: NLAClient,
    activations: list[torch.Tensor],
    contexts: list[str],
    *,
    num_tokens_per_sample: int = 5,
) -> ConfabulationReport:
    """Run full confabulation analysis on a set of activations.

    For each activation:
      1. Generate explanation
      2. Extract claims
      3. Verify each claim against context
      4. Measure claim recurrence (if multiple tokens available)
      5. Measure claim removal impact on MSE
    """
    all_claims: list[Claim] = []

    for idx, (activation, context) in enumerate(zip(activations, contexts)):
        explanation = client.generate_explanation(activation)
        claims = extract_claims(explanation)

        for claim in claims:
            claim.is_supported = verify_claim_against_context(claim, context)
            claim.mse_delta_on_removal = measure_claim_removal_impact(
                client, activation, explanation, claim
            )

        all_claims.extend(claims)

    # Aggregate statistics
    total = len(all_claims)
    supported = sum(1 for c in all_claims if c.is_supported is True)
    unsupported = sum(1 for c in all_claims if c.is_supported is False)
    unknown = sum(1 for c in all_claims if c.is_supported is None)

    def _rate(numerator: int, denominator: int) -> float:
        return numerator / max(1, denominator)

    thematic = [c for c in all_claims if c.specificity == "thematic"]
    entity = [c for c in all_claims if c.specificity == "entity"]
    detail = [c for c in all_claims if c.specificity == "detail"]
    recurring = [c for c in all_claims if c.recurrence_count > 0]

    true_deltas = [c.mse_delta_on_removal for c in all_claims
                   if c.is_supported is True and c.mse_delta_on_removal is not None]
    false_deltas = [c.mse_delta_on_removal for c in all_claims
                    if c.is_supported is False and c.mse_delta_on_removal is not None]

    return ConfabulationReport(
        total_claims=total,
        supported_claims=supported,
        unsupported_claims=unsupported,
        unknown_claims=unknown,
        support_rate=_rate(supported, supported + unsupported),
        thematic_support_rate=_rate(
            sum(1 for c in thematic if c.is_supported is True),
            sum(1 for c in thematic if c.is_supported is not None),
        ),
        entity_support_rate=_rate(
            sum(1 for c in entity if c.is_supported is True),
            sum(1 for c in entity if c.is_supported is not None),
        ),
        detail_support_rate=_rate(
            sum(1 for c in detail if c.is_supported is True),
            sum(1 for c in detail if c.is_supported is not None),
        ),
        recurring_claim_support_rate=_rate(
            sum(1 for c in recurring if c.is_supported is True),
            sum(1 for c in recurring if c.is_supported is not None),
        ),
        avg_mse_delta_true=sum(true_deltas) / max(1, len(true_deltas)),
        avg_mse_delta_false=sum(false_deltas) / max(1, len(false_deltas)),
        claims=all_claims,
    )
