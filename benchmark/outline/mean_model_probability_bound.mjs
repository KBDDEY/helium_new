/**
 * Exact combinatorial evaluation of a conservative non-regression certificate.
 *
 * This is a standalone mathematical calculator. It imports no project code,
 * starts no engine, and does not simulate actual task executions.
 *
 * Model and proof: p_priority_mean_model_bound.md.
 * In particular: P-first, preserve one card for every remaining P, no waiting,
 * cap(P) <= 4, every multi-card P takes <= 31s, and iid uniform I/D/P types.
 * A failed certificate is NOT necessarily a regression.
 *
 * Optional local/server usage: node mean_model_probability_bound.mjs [maxK]
 * Default maxK = 10. Work grows polynomially; this is not a large-K benchmark.
 */

export function choose(n, k) {
  if (k < 0 || k > n) return 0n;
  k = Math.min(k, n - k);
  let result = 1n;
  for (let j = 1; j <= k; j++) {
    result = result * BigInt(n - k + j) / BigInt(j);
  }
  return result;
}

// Lower bound on m + 2f when a cards are available for q remaining P.
export function releasedCapacityCertificate(a, q) {
  if (q < 1 || q > a) throw new Error("Expected 1 <= q <= a");
  const extraMax = Math.min(a - q, 3 * q);
  const remainingShortCards = a - q - extraMax;
  const multiCardPMin = Math.ceil(extraMax / 3);
  return multiCardPMin + 2 * remainingShortCards;
}

export function isCertified(K, r, B, D, p) {
  const I = r - B - D;
  const F = K - r;
  if (p === 0) return true;
  if (p <= F) {
    return 2 * B <= r + 1 + releasedCapacityCertificate(F, p);
  }
  if (p <= F + D) {
    return B <= 1 + 2 * I + releasedCapacityCertificate(D, p - F);
  }
  // Last P starts at or after t=15; proved safe independently of B.
  return true;
}

export function evaluate(K, r) {
  if (!Number.isInteger(K) || K < 5 || !Number.isInteger(r) || r < 1 || r > K - 3) {
    throw new Error("Expected K >= 5 and 1 <= r <= K-3");
  }
  const n0 = K + r - 1;
  let uncertified = 0n;
  let total = 0n;
  for (let B = 0; B <= r; B++) {
    const prefixPChoices = choose(r, B);
    for (let D = 0; D <= r - B; D++) {
      const prefixMultiplicity = prefixPChoices * choose(r - B, D);
      for (let p = 0; p <= K - 1; p++) {
        // Choose P positions; every other tail position is independently I/D.
        const tailMultiplicity = choose(K - 1, p) * 2n ** BigInt(K - 1 - p);
        const multiplicity = prefixMultiplicity * tailMultiplicity;
        total += multiplicity;
        if (!isCertified(K, r, B, D, p)) uncertified += multiplicity;
      }
    }
  }
  const denominator = 3n ** BigInt(n0);
  if (total !== denominator) throw new Error("Combinatorial mass mismatch");
  return { K, n0, r, uncertified, denominator };
}

export function worstForK(K) {
  let worst = null;
  for (let r = 1; r <= K - 3; r++) {
    const candidate = evaluate(K, r);
    if (worst === null ||
        candidate.uncertified * worst.denominator > worst.uncertified * candidate.denominator) {
      worst = candidate;
    }
  }
  return worst;
}

// Keep the exact rational result authoritative; the decimal is for display.
const maxK = process.argv[2] === undefined ? 10 : Number(process.argv[2]);
if (!Number.isInteger(maxK) || maxK < 5) {
  throw new Error("maxK must be an integer >= 5");
}
const rows = [];
for (let K = 5; K <= maxK; K++) {
  const value = worstForK(K);
  rows.push({
    K,
    largestBoundAtN0: value.n0,
    uncertifiedTypeSequences: value.uncertified.toString(),
    totalTypeSequences: value.denominator.toString(),
    regressionProbabilityUpperBoundPercent:
      100 * Number(value.uncertified) / Number(value.denominator),
  });
}
console.log(JSON.stringify({
  meaning: "Conservative probability upper bound, not measured or exact regression frequency",
  assumptions: "See p_priority_mean_model_bound.md",
  rows,
}, null, 2));
