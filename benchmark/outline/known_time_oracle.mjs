/**
 * Standalone, exact small-instance scheduling model (no project imports).
 *
 * All jobs are ready at time zero. A job chooses one mode before it starts,
 * occupies that many identical GPUs for its entire duration, and cannot be
 * preempted, resized, or migrated. Job order is unrestricted. Durations below
 * are deterministic TYPE-MEAN MODEL values, in integer milliseconds; they are
 * not predictions or bounds for actual individual Helium jobs.
 *
 * Only measured P modes 1--3 are included. P on four GPUs is UNKNOWN here.
 * D modes 1--3 are optional; I has only mode 1.
 *
 * Run independently, if desired: node benchmark/outline/known_time_oracle.mjs
 * This file does not import, start, or test Helium, vLLM, or the repository.
 */

export const PROFILES_MS = Object.freeze({
  P: Object.freeze({ 1: 50347, 2: 31056, 3: 24559 }),
  I: Object.freeze({ 1: 15000 }),
  D: Object.freeze({ 1: 10500, 2: 8500, 3: 7900 }),
});

function permutations(values) {
  if (values.length < 2) return [values.slice()];
  return values.flatMap((value, index) =>
    permutations(values.filter((_, other) => other !== index))
      .map(rest => [value, ...rest]));
}

function simulateList(cards, jobs) {
  let now = 0;
  let free = cards;
  let next = 0;
  let running = [];
  const schedule = [];
  while (next < jobs.length || running.length) {
    while (next < jobs.length && jobs[next].cards <= free) {
      const job = jobs[next++];
      const item = { ...job, startMs: now, endMs: now + job.durationMs };
      free -= item.cards;
      running.push(item);
      schedule.push(item);
    }
    if (!running.length) {
      if (next === jobs.length) break;
      throw new Error('Infeasible job width');
    }
    now = Math.min(...running.map(job => job.endMs));
    const remaining = [];
    for (const job of running) {
      if (job.endMs === now) free += job.cards;
      else remaining.push(job);
    }
    running = remaining;
  }
  return { makespanMs: now, schedule };
}

/**
 * Exhaustive event-time branch-and-bound for small independent-job instances.
 *
 * Completeness: with all jobs initially ready and no precedence constraints,
 * some optimal schedule starts jobs only at zero or completion events. Jobs
 * starting inside an interval with no completion can be moved to its left
 * endpoint (possibly together with other starts). Every feasible multiset of
 * mode starts is enumerated at each event, INCLUDING choosing not to start a
 * job although some GPUs are free. Thus useful deliberate idling is included.
 *
 * Pruning uses only feasible incumbents, completion/deadline checks, remaining
 * GPU-time lower bounds, and translation-equivalent state dominance. If the
 * node/time limit is hit, the result is feasible but NOT certified optimal.
 */
export function solveKnownTimes({
  cards = 6,
  counts = {},
  profiles = PROFILES_MS,
  allowed = {},
  timeLimitMs = 20000,
  maxNodes = 2000000,
} = {}) {
  if (!Number.isInteger(cards) || cards < 1) throw new Error('Invalid cards');
  if (Object.values(counts).some(count => !Number.isInteger(count) || count < 0))
    throw new Error('Counts must be nonnegative integers');
  if (Object.entries(counts).some(([type, count]) => count > 0 && !profiles[type]))
    throw new Error('Missing profile for a task type');
  const types = Object.keys(profiles).filter(type => (counts[type] ?? 0) > 0);
  const initial = types.map(type => counts[type]);
  if (!types.length) return {
    makespanMs: 0, schedule: [], provenOptimal: true, nodes: 0,
    memoStates: 0, lowerBoundMs: 0, stopReason: null,
  };
  const modesByType = types.map((type, typeIndex) => {
    const widths = allowed[type] ?? Object.keys(profiles[type]).map(Number);
    const modes = widths.filter(width => width <= cards).map(width => ({
      type, typeIndex, cards: width, durationMs: profiles[type][width],
    }));
    if (!modes.length || modes.some(mode => !Number.isInteger(mode.cards)
      || mode.cards < 1 || !Number.isInteger(mode.durationMs) || mode.durationMs <= 0))
      throw new Error(`Invalid modes for ${type}`);
    return modes.sort((a, b) => a.cards - b.cards);
  });
  const modes = modesByType.flat();

  let incumbent = null;
  function offerSeed(order, widths) {
    const jobs = [];
    for (const typeIndex of order) {
      const type = types[typeIndex];
      for (let j = 0; j < initial[typeIndex]; j++) {
        const width = Array.isArray(widths[typeIndex])
          ? widths[typeIndex][j] : widths[typeIndex];
        const mode = modesByType[typeIndex].find(item => item.cards === width);
        if (!mode) return;
        jobs.push({ type, id: `${type}${j + 1}`, cards: width,
          durationMs: mode.durationMs });
      }
    }
    const seed = simulateList(cards, jobs);
    if (!incumbent || seed.makespanMs < incumbent.makespanMs) incumbent = seed;
  }
  function seedWidths(typeIndex, widths, order) {
    if (typeIndex === types.length) return offerSeed(order, widths);
    for (const mode of modesByType[typeIndex])
      seedWidths(typeIndex + 1, [...widths, mode.cards], order);
  }
  const typeOrders = permutations(types.map((_, index) => index));
  for (const order of typeOrders) seedWidths(0, [], order);

  // Additional feasible seeds: full single-card P prefix, wider P tail.
  const pIndex = types.indexOf('P');
  if (pIndex >= 0 && modesByType[pIndex].some(mode => mode.cards === 1)) {
    const pCount = initial[pIndex];
    const remainder = pCount % cards;
    if (remainder > 0) {
      for (const mode of modesByType[pIndex]) {
        if (mode.cards * remainder > cards) continue;
        const widths = modesByType.map(items => items[0].cards);
        widths[pIndex] = Array(pCount - remainder).fill(1)
          .concat(Array(remainder).fill(mode.cards));
        for (const order of typeOrders.filter(order => order[0] === pIndex))
          offerSeed(order, widths);
      }
    }
  }

  let best = incumbent.makespanMs;
  let bestSchedule = incumbent.schedule;
  let nodes = 0;
  let stopReason = null;
  const deadline = Date.now() + timeLimitMs;
  const memo = new Map();
  const combinationsMemo = new Map();
  const trace = [];

  function startCombinations(free, remaining, canWait) {
    const key = `${free}|${remaining.map(count => Math.min(count, free)).join(',')}|${+canWait}`;
    if (combinationsMemo.has(key)) return combinationsMemo.get(key);
    const out = [];
    const usedByType = remaining.map(() => 0);
    const selected = [];
    function visit(index, capacity, total) {
      if (index === modes.length) {
        if (total || canWait) out.push({ starts: selected.slice(), count: total,
          used: free - capacity });
        return;
      }
      const mode = modes[index];
      const upper = Math.min(remaining[mode.typeIndex] - usedByType[mode.typeIndex],
        Math.floor(capacity / mode.cards));
      for (let count = 0; count <= upper; count++) {
        usedByType[mode.typeIndex] += count;
        for (let j = 0; j < count; j++) selected.push(mode);
        visit(index + 1, capacity - count * mode.cards, total + count);
        selected.length -= count;
        usedByType[mode.typeIndex] -= count;
      }
    }
    visit(0, free, 0);
    out.sort((a, b) => b.used - a.used || b.count - a.count);
    combinationsMemo.set(key, out);
    return out;
  }

  function search(now, remaining, running) {
    if (stopReason) return;
    nodes++;
    if (nodes > maxNodes) { stopReason = 'node limit'; return; }
    if ((nodes & 1023) === 0 && Date.now() > deadline) {
      stopReason = 'time limit'; return;
    }
    const last = Math.max(now, ...running.map(job => job.endMs));
    if (remaining.every(count => count === 0)) {
      if (last < best) { best = last; bestSchedule = trace.slice(); }
      return;
    }
    if (last >= best) return;
    let work = running.reduce((sum, job) => sum + job.cards * (job.endMs - now), 0);
    for (let t = 0; t < remaining.length; t++) {
      if (!remaining[t]) continue;
      const feasibleModes = modesByType[t].filter(mode => now + mode.durationMs < best);
      if (!feasibleModes.length) return;
      work += remaining[t] * Math.min(...feasibleModes.map(mode => mode.cards * mode.durationMs));
    }
    if (now + work / cards >= best) return;
    const relative = running.map(job => [job.endMs - now, job.cards])
      .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const key = `${remaining.join(',')}|${relative.map(pair => pair.join(':')).join(';')}`;
    if (memo.has(key) && memo.get(key) <= now) return;
    memo.set(key, now);
    const free = cards - running.reduce((sum, job) => sum + job.cards, 0);
    for (const combination of startCombinations(free, remaining, running.length > 0)) {
      const nextRemaining = remaining.slice();
      const nextRunning = running.slice();
      const priorTraceLength = trace.length;
      for (const mode of combination.starts) {
        const index = mode.typeIndex;
        const id = `${mode.type}${initial[index] - nextRemaining[index] + 1}`;
        nextRemaining[index]--;
        const job = { type: mode.type, id, cards: mode.cards,
          durationMs: mode.durationMs, startMs: now, endMs: now + mode.durationMs };
        nextRunning.push(job);
        trace.push(job);
      }
      const nextTime = Math.min(...nextRunning.map(job => job.endMs));
      search(nextTime, nextRemaining, nextRunning.filter(job => job.endMs !== nextTime));
      trace.length = priorTraceLength;
      if (stopReason) break;
    }
  }
  search(0, initial, []);
  const workLowerBound = initial.reduce((sum, count, index) => sum + count
    * Math.min(...modesByType[index].map(mode => mode.cards * mode.durationMs)), 0) / cards;
  const longestShortestJob = Math.max(...modesByType.map(items =>
    Math.min(...items.map(mode => mode.durationMs))));
  return {
    cards, counts: Object.fromEntries(types.map((type, index) => [type, initial[index]])),
    makespanMs: best, schedule: bestSchedule,
    provenOptimal: stopReason === null, nodes, memoStates: memo.size,
    lowerBoundMs: Math.ceil(Math.max(workLowerBound, longestShortestJob)),
    certifiedLowerBoundMs: stopReason === null ? best : null,
    certificate: stopReason === null ? 'complete event enumeration with valid pruning' : null,
    stopReason,
  };
}

export const EXAMPLE_CASES = Object.freeze([
  { label: '6 GPUs, 8 P', cards: 6, counts: { P: 8 }, expectedMs: 74906 },
  { label: '6 GPUs, 8 P + 8 I', cards: 6, counts: { P: 8, I: 8 }, expectedMs: 96403 },
  { label: '6 GPUs, 3 P + 6 I', cards: 6, counts: { P: 3, I: 6 }, expectedMs: 46056 },
  { label: '6 GPUs, 4 P + 4 I', cards: 6, counts: { P: 4, I: 4 }, expectedMs: 50347 },
  { label: '6 GPUs, 1 P + 27 I', cards: 6, counts: { P: 1, I: 27 }, expectedMs: 80347 },
  { label: '6 GPUs, 2 P + 6 D + 3 I', cards: 6,
    counts: { P: 2, D: 6, I: 3 }, expectedMs: 41556 },
]);

export function runExamples() {
  return EXAMPLE_CASES.map(example => ({ label: example.label,
    expectedMs: example.expectedMs, ...solveKnownTimes(example) }));
}

if (typeof process !== 'undefined' && /known_time_oracle\.mjs$/.test(process.argv[1] ?? '')) {
  console.log(JSON.stringify(runExamples(), null, 2));
}
