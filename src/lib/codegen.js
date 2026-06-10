// Parallel R / Python code generation (Task E). Pure functions that turn the LIVE
// sampler config into runnable code mirroring the simulation, so students can learn to
// write it themselves. Every generator reads the *same* specs the UI uses (stage outcomes,
// the `computeStat` fn semantics, the run-until rule) so the code and the tool never diverge.
//
// Each generator returns an array of LINES — `{ text, section }` — where `section` is one of
// "sampler" | "single" | "collect" | "inference". The four panels render the text beside the
// tool they mirror; the lines read top-to-bottom as one program (imports appear once, in the
// Sampler section).
//
// TWO shapes of output, picked by `isSimple`:
//   • COMPACT (the common teaching case): one stage, no fork, fixed n, a single-column
//     univariate statistic. We draw the whole sample in one call and inline the statistic —
//     no helper functions. e.g. R `sample(c(1:10), 10, replace=TRUE)` → `mean(x)`.
//   • SPLIT (forks, run-until, or multi-column stats like slope / group means): keeps the
//     `draw_one` / `draw_sample` / `compute_stat` decomposition those genuinely require.
//
// Python uses pandas/numpy (and statsmodels for regression); R is base R. Scope (v1): sampling
// is WITH replacement (a without-replacement device is flagged in a comment, not reproduced);
// population SD; type-7 quantiles. The headline statistic is the first plain tracked stat (or a
// sensible default); derived columns aren't emitted.

import { stageVarKind, stageOutcomes } from "./sampling";

// ─── Literals & identifiers ───────────────────────────────────────────────────
// A label is emitted as a numeric literal only when it parses as a finite number — so a
// numeric variable's draws are numbers (mean/SD work) while "a"/"8:30"/etc. stay quoted.
const isNumLit = v => { const s = String(v).trim(); return s !== "" && !isNaN(Number(s)); };
const lit = v => (isNumLit(v) ? String(Number(v)) : JSON.stringify(String(v)));
// JSON.stringify gives a safely-escaped double-quoted string, valid in both R and Python.
const key = name => JSON.stringify(name);
const vec = (labels, lang) => (lang === "r" ? "c(" : "[") + labels.map(lit).join(", ") + (lang === "r" ? ")" : "]");
// Sanitize a varName into a valid R/Python identifier; dedupe across the pipeline so two
// names that collapse to the same identifier ("a b" / "a-b") still get distinct symbols.
function safeName(raw) {
  let s = String(raw || "").replace(/[^A-Za-z0-9_]/g, "_");
  if (!s) s = "v";
  if (/^[0-9]/.test(s)) s = "v" + s;
  return s;
}
function buildNames(pipeline) {
  const map = {}, used = new Set();
  pipeline.forEach(st => {
    const base = safeName(st.varName);
    let n = base, k = 2;
    while (used.has(n)) n = base + "_" + k++;
    used.add(n); map[st.id] = n;
  });
  return map;
}

// ─── A device → its outcome space (labels + weights) ──────────────────────────
// Spinner weights = slice pct; stacks weights = item counts; mixer = distinct labels weighted
// by ball multiplicity. `uniform` lets us drop the weights argument when every weight is equal.
function outcomeSpec(dev) {
  const labels = [], weights = [];
  if (dev.type === "spinner") dev.slices.forEach(s => { labels.push(s.label); weights.push(s.pct); });
  else if (dev.type === "stacks") dev.items.forEach(it => { labels.push(it.label); weights.push(it.count); });
  else if (dev.type === "mixer") {
    const at = {};
    dev.balls.forEach(b => {
      if (!(b.label in at)) { at[b.label] = labels.length; labels.push(b.label); weights.push(0); }
      weights[at[b.label]]++;
    });
  }
  const uniform = weights.every(w => w === weights[0]);
  const replace = dev.type === "spinner" ? true : dev.withReplacement !== false;
  return { labels, weights, uniform, replace };
}
// A stage with a single (default) branch — the device it always draws from.
const defDevice = st => (st.branches.find(b => b.condVar === null) || st.branches[0]).device;

// ─── Region predicate for countBetween / propBetween (mirrors computeStat's inR) ─
// Elementwise on a vector/Series in both languages, so `&` (not R's `&&` / Python's `and`).
function regionExpr(xExpr, s, lang) {
  const num = v => String(parseFloat(Number(v).toFixed(4)));
  const parts = [];
  if (s.lo != null) parts.push(`${xExpr} ${s.loOpen ? ">" : ">="} ${num(s.lo)}`);
  if (s.hi != null) parts.push(`${xExpr} ${s.hiOpen ? "<" : "<="} ${num(s.hi)}`);
  if (!parts.length) return lang === "r" ? "TRUE" : "True";
  return parts.length > 1 ? parts.map(p => `(${p})`).join(" & ") : parts[0];
}

// Choose the headline statistic: the first plain (non-derived) tracked stat, else a sensible
// default — mean of the first numeric stage, or the proportion of the first outcome.
function headlineStat(cfg) {
  const plain = (cfg.trackedStats || []).find(s => s && s.kind !== "derived" && s.fn);
  if (plain) return plain;
  const numStage = cfg.pipeline.find(st => stageVarKind(st).numeric);
  if (numStage) return { fn: "mean", variable: numStage.id };
  const st = cfg.pipeline[0];
  return st ? { fn: "proportion", variable: st.id, target: stageOutcomes(st)[0] || "" } : { fn: "count", variable: "" };
}
// Name the collected sampling-distribution vector after the statistic (matches the example:
// means / medians / proportions / slopes …).
function resultName(s) {
  const m = { mean: "means", median: "medians", sd: "sds", proportion: "proportions", countVal: "counts",
    count: "counts", slope: "slopes", intercept: "intercepts", q1: "q1s", q3: "q3s", min: "mins",
    max: "maxes", propBetween: "proportions", countBetween: "counts" };
  return (s && m[s.fn]) || "stats";
}
// A statistic is single-column when it reads one column and needs no row alignment.
const singleColumn = s => !s.condVar && !s.variable2 && s.fn !== "slope" && s.fn !== "intercept";

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  COMPACT PATH — one stage, no fork, fixed n, single-column univariate stat     ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
// One sample drawn in a single call; the statistic inlined. No helper functions.

// The whole-sample draw expression for one device.
function drawVec(col, dev, lang) {
  const { labels, weights, uniform } = outcomeSpec(dev);
  if (!labels.length) return lang === "r" ? `${col} <- NA` : `${col} = pop_${col}.sample(n, replace=True)`;
  if (lang === "r") {
    if (labels.length === 1) return `${col} <- rep(${lit(labels[0])}, n)`;   // avoids R's sample(c(5),n) 1:5 trap
    const prob = uniform ? "" : `, prob = c(${weights.join(", ")})`;
    return `${col} <- sample(${vec(labels, "r")}, n, replace = TRUE${prob})`;
  }
  const w = uniform ? "" : `, weights=[${weights.join(", ")}]`;
  return `${col} = pop_${col}.sample(n, replace=True${w})`;
}
// The inline statistic expression over the drawn column `v` (an R vector / a pandas Series).
function vStat(s, v, lang) {
  if (lang === "r") {
    switch (s.fn) {
      case "mean": return `mean(${v})`;
      case "median": return `median(${v})`;
      case "sd": return `sqrt(mean((${v} - mean(${v}))^2))`;
      case "min": return `min(${v})`;
      case "max": return `max(${v})`;
      case "q1": return `as.numeric(quantile(${v}, 0.25, type = 7))`;
      case "q3": return `as.numeric(quantile(${v}, 0.75, type = 7))`;
      case "count": return `length(${v})`;
      case "countVal": return `sum(${v} == ${lit(s.target)})`;
      case "proportion": return `mean(${v} == ${lit(s.target)})`;
      case "countBetween": return `sum(${regionExpr(v, s, "r")})`;
      case "propBetween": return `mean(${regionExpr(v, s, "r")})`;
      default: return "NA";
    }
  }
  switch (s.fn) {
    case "mean": return `${v}.mean()`;
    case "median": return `${v}.median()`;
    case "sd": return `${v}.std(ddof=0)`;
    case "min": return `${v}.min()`;
    case "max": return `${v}.max()`;
    case "q1": return `${v}.quantile(0.25)`;
    case "q3": return `${v}.quantile(0.75)`;
    case "count": return `len(${v})`;
    case "countVal": return `(${v} == ${lit(s.target)}).sum()`;
    case "proportion": return `(${v} == ${lit(s.target)}).mean()`;
    case "countBetween": return `(${regionExpr(v, s, "py")}).sum()`;
    case "propBetween": return `(${regionExpr(v, s, "py")}).mean()`;
    default: return "float('nan')";
  }
}

function genCompact(cfg, names, lang) {
  const st = cfg.pipeline[0], dev = defDevice(st);
  const col = names[st.id];
  const s = headlineStat(cfg), v = names[s.variable] || col;
  const R = resultName(s);
  const { labels, replace } = outcomeSpec(dev);
  const wor = replace === false;
  const mk = section => { const a = []; a.push = t => Array.prototype.push.call(a, { text: t, section }); return a; };

  if (lang === "r") {
    const sampler = mk("sampler");
    sampler.push("# Sampler — draw one sample of n from the device");
    if (wor) sampler.push("# (the device draws without replacement; this code samples with replacement)");
    sampler.push(`n <- ${cfg.sampleSize}`);
    sampler.push(drawVec(col, dev, "r"));

    const single = mk("single");
    single.push("# The statistic for one sample");
    single.push(`stat <- ${vStat(s, v, "r")}`);

    const collect = mk("collect");
    collect.push("# Collect the statistic over many samples");
    collect.push("N <- 1000");
    collect.push(`${R} <- numeric(N)`);
    collect.push("for (i in 1:N) {");
    collect.push("  " + drawVec(col, dev, "r"));
    collect.push(`  ${R}[i] <- ${vStat(s, v, "r")}`);
    collect.push("}");

    const inference = mk("inference");
    inference.push("# Inference from the sampling distribution");
    inference.push(`quantile(${R}, c(0.025, 0.975))   # 95% percentile interval`);
    inference.push(`mean(${R} >= 0)                   # tail proportion — set your cutoff`);
    return { sampler, single, collect, inference };
  }

  // Python (pandas / numpy)
  const sampler = mk("sampler");
  sampler.push("import pandas as pd");
  sampler.push("import numpy as np");
  sampler.push("");
  sampler.push("# Sampler — draw one sample of n from the device");
  if (wor) sampler.push("# (the device draws without replacement; this code samples with replacement)");
  sampler.push(`n = ${cfg.sampleSize}`);
  sampler.push(`pop_${col} = pd.Series(${vec(labels, "py")})`);
  sampler.push(drawVec(col, dev, "py"));

  const single = mk("single");
  single.push("# The statistic for one sample");
  single.push(`stat = ${vStat(s, v, "py")}`);

  const collect = mk("collect");
  collect.push("# Collect the statistic over many samples");
  collect.push("N = 1000");
  collect.push(`${R} = []`);
  collect.push("for i in range(N):");
  collect.push("    " + drawVec(col, dev, "py"));
  collect.push(`    ${R}.append(${vStat(s, v, "py")})`);

  const inference = mk("inference");
  inference.push("# Inference from the sampling distribution");
  inference.push(`${R} = np.array(${R})`);
  inference.push(`np.quantile(${R}, [0.025, 0.975])   # 95% percentile interval`);
  inference.push(`(${R} >= 0).mean()                  # tail proportion — set your cutoff`);
  return { sampler, single, collect, inference };
}

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  SPLIT PATH — forks, run-until, or multi-column stats need per-row drawing      ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// One device → a single-draw expression (per-row). Length-1 outcome → the literal itself
// (deterministic, and sidesteps R's `sample(c(5), 1)` "sample from 1:5" trap).
function weighted(labels, w, lang) {
  const allEq = w.every(x => x === w[0]);
  if (lang === "r") return allEq ? `sample(${vec(labels, "r")}, 1)` : `sample(${vec(labels, "r")}, 1, prob = c(${w.join(", ")}))`;
  return allEq ? `random.choices(${vec(labels, "py")})[0]` : `random.choices(${vec(labels, "py")}, weights = [${w.join(", ")}])[0]`;
}
function deviceDraw(dev, lang) {
  const { labels, weights } = outcomeSpec(dev);
  if (!labels.length) return lang === "r" ? "NA" : "None";
  if (labels.length === 1) return lit(labels[0]);
  return weighted(labels, weights, lang);
}
// One stage → an assignment block, mirroring `selectBranch` (conditional branches in order,
// then the default `else`).
function stageBlock(stage, names, lang) {
  const name = names[stage.id];
  const conds = stage.branches.filter(b => b.condVar !== null);
  const def = stage.branches.find(b => b.condVar === null) || stage.branches[0];
  if (!conds.length) {
    return [lang === "r" ? `  ${name} <- ${deviceDraw(def.device, "r")}` : `    ${name} = ${deviceDraw(def.device, "py")}`];
  }
  if (lang === "r") {
    const out = [];
    conds.forEach((b, i) => {
      const head = i === 0 ? "if" : "} else if";
      out.push(`  ${i === 0 ? name + " <- " : ""}${head} (${names[b.condVar]} == ${lit(b.condVal)}) {`);
      out.push(`    ${deviceDraw(b.device, "r")}`);
    });
    out.push(`  } else {`);
    out.push(`    ${deviceDraw(def.device, "r")}  # otherwise`);
    out.push(`  }`);
    return out;
  }
  const out = [];
  conds.forEach((b, i) => {
    out.push(`    ${i === 0 ? "if" : "elif"} ${names[b.condVar]} == ${lit(b.condVal)}:`);
    out.push(`        ${name} = ${deviceDraw(b.device, "py")}`);
  });
  out.push(`    else:`);
  out.push(`        ${name} = ${deviceDraw(def.device, "py")}  # otherwise`);
  return out;
}
// Run-until stop predicate (mirrors `stopReached`).
function stopPredicate(rule, names, src, lang) {
  const col = names[rule.stageId], n = rule.n || 1;
  if (lang === "r") {
    const c = `${src}$${col}`;
    if (rule.kind === "outcome") return `any(${c} == ${lit(rule.value)})`;
    if (rule.kind === "count") return `sum(${c} == ${lit(rule.value)}) >= ${n}`;
    return `length(unique(${c})) >= ${n}`;
  }
  const c = `r[${key(col)}]`;
  if (rule.kind === "outcome") return `any(${c} == ${lit(rule.value)} for r in ${src})`;
  if (rule.kind === "count") return `sum(1 for r in ${src} if ${c} == ${lit(rule.value)}) >= ${n}`;
  return `len(set(${c} for r in ${src})) >= ${n}`;
}

// R: the trailing expression of compute_stat. Python: its body lines after `sub`.
function rStatExpr(s, v, v2) {
  const x = `sub$${v}`;
  switch (s.fn) {
    case "count": return "nrow(sub)";
    case "countVal": return `sum(${x} == ${lit(s.target)})`;
    case "proportion": return `mean(${x} == ${lit(s.target)})`;
    case "mean": return `mean(${x})`;
    case "sd": return `sqrt(mean((${x} - mean(${x}))^2))`;
    case "median": return `median(${x})`;
    case "min": return `min(${x})`;
    case "max": return `max(${x})`;
    case "q1": return `as.numeric(quantile(${x}, 0.25, type = 7))`;
    case "q3": return `as.numeric(quantile(${x}, 0.75, type = 7))`;
    case "slope": return `unname(coef(lm(${v2} ~ ${v}, data = sub))[2])`;
    case "intercept": return `unname(coef(lm(${v2} ~ ${v}, data = sub))[1])`;
    case "countBetween": return `sum(${regionExpr(x, s, "r")})`;
    case "propBetween": return `mean(${regionExpr(x, s, "r")})`;
    default: return "NA";
  }
}
function pyStatExpr(s, v, v2) {
  const c = `sub[${key(v)}]`;
  const ret = e => [`    return ${e}`];
  switch (s.fn) {
    case "count": return ret("len(sub)");
    case "countVal": return ret(`(${c} == ${lit(s.target)}).sum()`);
    case "proportion": return ret(`(${c} == ${lit(s.target)}).mean()`);
    case "mean": return ret(`${c}.mean()`);
    case "sd": return ret(`${c}.std(ddof=0)`);
    case "median": return ret(`${c}.median()`);
    case "min": return ret(`${c}.min()`);
    case "max": return ret(`${c}.max()`);
    case "q1": return ret(`${c}.quantile(0.25)`);
    case "q3": return ret(`${c}.quantile(0.75)`);
    case "slope": return [`    fit = sm.OLS(sub[${key(v2)}], sm.add_constant(sub[${key(v)}])).fit()`, "    return fit.params.iloc[1]"];
    case "intercept": return [`    fit = sm.OLS(sub[${key(v2)}], sm.add_constant(sub[${key(v)}])).fit()`, "    return fit.params.iloc[0]"];
    case "countBetween": return ret(`(${regionExpr(c, s, "py")}).sum()`);
    case "propBetween": return ret(`(${regionExpr(c, s, "py")}).mean()`);
    default: return ret("float('nan')");
  }
}

function genSplit(cfg, names, lang) {
  const { pipeline, sampleSize, runMode, stopRule } = cfg;
  const until = runMode === "until" && stopRule && stopRule.stageId;
  const cap = until ? "max_draws" : "n";
  const wor = pipeline.some(st => st.branches.some(b => b.device.withReplacement === false));
  const s = headlineStat(cfg);
  const v = names[s.variable], v2 = names[s.variable2], cond = names[s.condVar];
  const needsSm = s.fn === "slope" || s.fn === "intercept";
  const R = resultName(s);
  const mk = section => { const a = []; a.push = t => Array.prototype.push.call(a, { text: t, section }); return a; };

  if (lang === "r") {
    const sampler = mk("sampler");
    sampler.push("# Sampler — draw one row through the pipeline");
    if (wor) sampler.push("# (a device draws without replacement in the tool; this code samples with replacement)");
    sampler.push("draw_one <- function() {");
    pipeline.forEach(st => stageBlock(st, names, "r").forEach(t => sampler.push(t)));
    sampler.push("  data.frame(" + pipeline.map(st => `${names[st.id]} = ${names[st.id]}`).join(", ") + ", stringsAsFactors = FALSE)");
    sampler.push("}");
    sampler.push("");
    if (until) {
      sampler.push("# Draw rows until the stop rule holds (n varies), capped at max_draws");
      sampler.push("draw_sample <- function(max_draws) {");
      sampler.push("  rows <- list()");
      sampler.push("  repeat {");
      sampler.push("    rows[[length(rows) + 1]] <- draw_one()");
      sampler.push("    df <- do.call(rbind, rows)");
      sampler.push(`    if (${stopPredicate(stopRule, names, "df", "r")}) break`);
      sampler.push("    if (length(rows) >= max_draws) break");
      sampler.push("  }");
      sampler.push("  do.call(rbind, rows)");
      sampler.push("}");
    } else {
      sampler.push("# A sample = n rows drawn through the pipeline");
      sampler.push("draw_sample <- function(n) do.call(rbind, lapply(seq_len(n), function(i) draw_one()))");
    }

    const single = mk("single");
    single.push(`${cap} <- ${sampleSize}` + (until ? "   # safety cap (max draws)" : "   # sample size"));
    single.push("compute_stat <- function(df) {");
    single.push(s.condVar ? `  sub <- df[df$${cond} == ${lit(s.condVal)}, , drop = FALSE]` : "  sub <- df");
    single.push("  " + rStatExpr(s, v, v2));
    single.push("}");
    single.push(`stat <- compute_stat(draw_sample(${cap}))`);

    const collect = mk("collect");
    collect.push("# Collect the statistic over many samples");
    collect.push("N <- 1000");
    collect.push(`${R} <- replicate(N, compute_stat(draw_sample(${cap})))`);

    const inference = mk("inference");
    inference.push("# Inference from the sampling distribution");
    inference.push(`quantile(${R}, c(0.025, 0.975))   # 95% percentile interval`);
    inference.push(`mean(${R} >= 0)                   # tail proportion — set your cutoff`);
    return { sampler, single, collect, inference };
  }

  // Python (pandas / numpy / statsmodels)
  const sampler = mk("sampler");
  sampler.push("import random");
  sampler.push("import pandas as pd");
  sampler.push("import numpy as np");
  if (needsSm) sampler.push("import statsmodels.api as sm");
  sampler.push("");
  sampler.push("# Sampler — draw one row through the pipeline");
  if (wor) sampler.push("# (a device draws without replacement in the tool; this code samples with replacement)");
  sampler.push("def draw_one():");
  pipeline.forEach(st => stageBlock(st, names, "py").forEach(t => sampler.push(t)));
  sampler.push("    return {" + pipeline.map(st => `${key(names[st.id])}: ${names[st.id]}`).join(", ") + "}");
  sampler.push("");
  if (until) {
    sampler.push("# Draw rows until the stop rule holds (n varies), capped at max_draws");
    sampler.push("def draw_sample(max_draws):");
    sampler.push("    rows = []");
    sampler.push("    while True:");
    sampler.push("        rows.append(draw_one())");
    sampler.push(`        if ${stopPredicate(stopRule, names, "rows", "py")}: break`);
    sampler.push("        if len(rows) >= max_draws: break");
    sampler.push("    return pd.DataFrame(rows)");
  } else {
    sampler.push("# A sample = n rows drawn through the pipeline");
    sampler.push("def draw_sample(n):");
    sampler.push("    return pd.DataFrame([draw_one() for _ in range(n)])");
  }

  const single = mk("single");
  single.push(`${cap} = ${sampleSize}` + (until ? "   # safety cap (max draws)" : "   # sample size"));
  single.push("def compute_stat(df):");
  single.push(s.condVar ? `    sub = df[df[${key(cond)}] == ${lit(s.condVal)}]` : "    sub = df");
  pyStatExpr(s, v, v2).forEach(t => single.push(t));
  single.push(`stat = compute_stat(draw_sample(${cap}))`);

  const collect = mk("collect");
  collect.push("# Collect the statistic over many samples");
  collect.push("N = 1000");
  collect.push(`${R} = [compute_stat(draw_sample(${cap})) for _ in range(N)]`);

  const inference = mk("inference");
  inference.push("# Inference from the sampling distribution");
  inference.push(`${R} = np.array(${R})`);
  inference.push(`np.quantile(${R}, [0.025, 0.975])   # 95% percentile interval`);
  inference.push(`(${R} >= 0).mean()                  # tail proportion — set your cutoff`);
  return { sampler, single, collect, inference };
}

// ─── Top-level ────────────────────────────────────────────────────────────────
// COMPACT when: one stage, no fork, fixed n, and a single-column univariate headline stat.
function isSimple(cfg) {
  const p = cfg.pipeline || [];
  if (p.length !== 1) return false;
  if (cfg.runMode === "until") return false;
  const forked = st => st.branches.length > 1 || st.branches.some(b => b.condVar !== null);
  if (p.some(forked)) return false;
  return singleColumn(headlineStat(cfg));
}

export function generateCode(cfg, lang) {
  const language = lang === "python" ? "py" : "r";
  const names = buildNames(cfg.pipeline || []);
  const { sampler, single, collect, inference } =
    isSimple(cfg) ? genCompact(cfg, names, language) : genSplit(cfg, names, language);
  // Integrated = the four sections stitched into one program (kept for a possible single-program
  // view; the distributed layout uses the four sections directly).
  const integrated = [];
  [sampler, single, collect, inference].forEach((sec, i) => {
    if (i > 0) integrated.push({ text: "", section: sec[0] ? sec[0].section : "sampler" });
    sec.forEach(ln => integrated.push(ln));
  });
  return { sampler, single, collect, inference, integrated };
}
