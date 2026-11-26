import { useCallback, useEffect, useMemo, useState } from "react";

const CLASSES = [
  ["crazing", "Crazing", "Fine interconnected surface cracks"],
  ["inclusion", "Inclusion", "Embedded irregular inclusions"],
  ["patches", "Patches", "Irregular contrast patches"],
  ["pitted_surface", "Pitted surface", "Small surface pits"],
  ["rolled_in_scale", "Rolled-in scale", "Rolled-in mill scale"],
  ["scratches", "Scratches", "Long, narrow surface scratches"],
];
const ARMS = [
  ["real_only", "Real-only"],
  ["synthetic_only", "Synthetic-only"],
  ["real_plus_synthetic", "Real + synthetic"],
];
const METRICS = [
  ["accuracy", "Accuracy"],
  ["macro_precision", "Macro precision"],
  ["macro_recall", "Macro recall"],
  ["macro_f1", "Macro F1"],
];
const FIGURES = [
  ["02_accuracy_by_seed.png", "Accuracy by arm and seed"],
  ["03_macro_f1_by_seed.png", "Macro F1 by arm and seed"],
  ["04_precision_recall.png", "Precision and recall"],
  ["05_mean_sd_all_metrics.png", "Mean and sample standard deviation"],
  ["09_per_class_f1.png", "Per-class F1"],
  ["10_pairwise_macro_f1_differences.png", "Paired macro F1 differences"],
];
const CONFUSIONS = ARMS.map(([arm, label]) => [
  `06_confusion_${arm}.png`,
  `${label} · pooled confusion matrix`,
]);
const pretty = (value) => value?.toLocaleString?.() ?? value;
const pct = (value) => `${(100 * value).toFixed(2)}%`;
const armLabel = (value) => ARMS.find(([key]) => key === value)?.[1] ?? value;
const classLabel = (value) => CLASSES.find(([key]) => key === value)?.[1] ?? value;

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function Icon({ children }) {
  return <span className="icon-mark" aria-hidden="true">{children}</span>;
}

function Stat({ label, value, note, tone = "blue" }) {
  return (
    <article className={`stat-card stat-${tone}`}>
      <p className="eyebrow">{label}</p>
      <div className="stat-value">{value}</div>
      {note && <p className="stat-note">{note}</p>}
    </article>
  );
}

function Header({ page, setPage, health }) {
  const nav = [
    ["overview", "Overview", "01"],
    ["demo", "Live demo", "02"],
    ["results", "Results", "03"],
  ];
  return (
    <header className="topbar">
      <button className="brand" onClick={() => setPage("overview")} aria-label="SteelSight overview">
        <span className="brand-symbol"><span /><span /><span /></span>
        <span><strong>STEELSIGHT</strong><small>RESEARCH PRESENTATION</small></span>
      </button>
      <nav className="nav-tabs" aria-label="Main navigation">
        {nav.map(([key, title, number]) => (
          <button key={key} className={page === key ? "nav-item active" : "nav-item"} onClick={() => setPage(key)}>
            <span>{number}</span>{title}
          </button>
        ))}
      </nav>
      <div className="system-status">
        <span className={health?.cuda_available ? "status-dot live" : "status-dot"} />
        <span>{health?.gpu_name || "Checking local GPU"}</span>
        {health?.busy && <span className="busy-pill">GPU task running</span>}
      </div>
    </header>
  );
}

function SectionHeading({ kicker, title, children, action }) {
  return (
    <div className="section-heading">
      <div><p className="eyebrow">{kicker}</p><h2>{title}</h2>{children && <p className="section-copy">{children}</p>}</div>
      {action}
    </div>
  );
}

function Pipeline({ overview }) {
  const stages = [
    ["01", "Real steel", "NEU-CLS · frozen groups", "1,260 train"],
    ["02", "LoRA adaptation", "SD 1.5 · 1,000 updates", "Train split only"],
    ["03", "Synthetic pool", "996 generated · 783 selected", "186 near matches retained"],
    ["04", "ResNet-18", "Three arms · seeds 42–44", "Same real test set"],
  ];
  return (
    <div className="pipeline-wrap">
      <div className="pipeline-track">
        {stages.map(([n, title, detail, foot], index) => (
          <div className="pipeline-stage" key={n}>
            <div className={`pipeline-node node-${index}`}><span>{n}</span><b>{title}</b></div>
            <p>{detail}</p><small>{foot}</small>
            {index < stages.length - 1 && <div className="pipeline-connector">→</div>}
          </div>
        ))}
      </div>
      <div className="test-lane"><span className="lane-rule" /><span className="lane-chip">270 held-out real images · opened for final evaluation only</span><span className="lane-rule" /></div>
      {overview && <div className="pipeline-foot">Frozen manifest <code>{overview.manifest_sha256}</code></div>}
    </div>
  );
}

function OverviewPage({ overview, results, goDemo, goResults, loading }) {
  if (loading) return <Loading label="Loading project overview…" />;
  if (!overview || !results) return <ErrorPanel message="Project overview could not be loaded. Check that the backend is running." />;
  const baseline = results.aggregates.real_only;
  const synthetic = results.aggregates.synthetic_only;
  const mixed = results.aggregates.real_plus_synthetic;
  return (
    <main className="page-shell">
      <section className="hero hero-overview">
        <div className="hero-copy">
          <p className="eyebrow light">INDUSTRIAL COMPUTER VISION · RESEARCH STUDY</p>
          <h1>Can synthetic steel defects help models see the real thing?</h1>
          <p className="hero-subtitle">A reproducible comparison of generated steel-surface defects and ResNet-18 classification on a held-out real inspection set.</p>
          <div className="hero-actions"><button className="button button-light" onClick={goDemo}>Open live demo <span>↗</span></button><button className="button button-ghost" onClick={goResults}>Explore results</button></div>
          <div className="hero-tags"><span>Stable Diffusion 1.5</span><span>LoRA · 1,000 updates</span><span>NEU-CLS · 6 classes</span></div>
        </div>
        <div className="hero-art" aria-hidden="true"><div className="scan-ring ring-one"/><div className="scan-ring ring-two"/><div className="scan-core"><span>NEU</span><i>● ● ●</i><b>STEEL SURFACE</b></div><div className="scan-label label-a">DEFECT / 06</div><div className="scan-label label-b">VISION SYSTEM / ONLINE</div><div className="scan-axis"/></div>
      </section>

      <section className="stat-grid">
        <Stat label="Real-only mean accuracy" value={pct(baseline.accuracy.mean)} note="All three seeds reached 100%" tone="blue" />
        <Stat label="Synthetic-only mean macro F1" value={pct(synthetic.macro_f1.mean)} note="Synthetic-to-real transfer" tone="amber" />
        <Stat label="Real + synthetic mean macro F1" value={pct(mixed.macro_f1.mean)} note="No gain over the real-only ceiling" tone="green" />
        <Stat label="Selected synthetic images" value={pretty(overview.selected_synthetic)} note="Automated eligibility policy" tone="slate" />
      </section>

      <section className="content-section">
        <SectionHeading kicker="RESEARCH PIPELINE" title="From training data to held-out evaluation">Each experiment uses the same frozen real split. Test samples are reserved for the final comparison.</SectionHeading>
        <Pipeline overview={overview} />
      </section>

      <section className="two-column">
        <article className="panel panel-padding">
          <SectionHeading kicker="THE EXPERIMENT" title="Three training sources" />
          <div className="arm-list">
            {ARMS.map(([arm, label], i) => <div className="arm-row" key={arm}><span className={`arm-index arm-${i}`}>0{i + 1}</span><div><strong>{label}</strong><p>{arm === "real_only" ? "Frozen NEU-CLS training images only." : arm === "synthetic_only" ? "Automated-selection synthetic pool only." : "Real train plus the selected synthetic pool; 50:50 sampling."}</p></div><span className="arm-arrow">↗</span></div>)}
          </div>
        </article>
        <article className="panel finding-panel">
          <p className="eyebrow">CENTRAL FINDING</p><div className="finding-mark">↘</div>
          <h3>More generated data did not lift an already-perfect baseline.</h3>
          <p>Real-only averaged 100% on this test set. The mixed arm averaged {pct(mixed.macro_f1.mean)} macro F1, while synthetic-only averaged {pct(synthetic.macro_f1.mean)}.</p>
          <button className="text-link" onClick={goResults}>See the full comparison <span>→</span></button>
        </article>
      </section>

      <section className="content-section">
        <SectionHeading kicker="METHODOLOGY AT A GLANCE" title="A fixed split, a shared classifier, paired seeds" />
        <div className="method-grid">
          <div className="method-card"><span className="method-number">01</span><h3>Dataset</h3><p>NEU-CLS steel surface dataset. Six canonical classes, original grayscale images preserved.</p><strong>1,260 / 270 / 270</strong><small>train / validation / test</small></div>
          <div className="method-card"><span className="method-number">02</span><h3>Generator</h3><p>Stable Diffusion 1.5 with an attention LoRA adapter trained on real training images.</p><strong>1,000 updates</strong><small>pinned local model snapshot</small></div>
          <div className="method-card"><span className="method-number">03</span><h3>Classifier</h3><p>ImageNet-initialized ResNet-18 with a six-class head; full backbone and head trainable.</p><strong>1,000 updates</strong><small>AdamW · cosine schedule</small></div>
          <div className="method-card"><span className="method-number">04</span><h3>Evaluation</h3><p>All nine checkpoints scored against the exact same real test image IDs.</p><strong>3 paired seeds</strong><small>42 · 43 · 44</small></div>
        </div>
      </section>

      <section className="notice notice-amber"><span>!</span><div><strong>Selection status matters</strong><p>{overview.selected_synthetic} images passed automated blocker/warning filtering. They were not individually human-approved. {overview.selected_near_matches} retained images have near-match relationships. Automated eligibility is not a visual quality verdict.</p></div></section>
    </main>
  );
}

function DemoPage({ health, models, gallery, onGenerate, onClassifyUpload, onClassifyGenerated, busy }) {
  const [className, setClassName] = useState("patches");
  const [arm, setArm] = useState("real_only");
  const [seed, setSeed] = useState(42);
  const [image, setImage] = useState(null);
  const [uploaded, setUploaded] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [seedValue, setSeedValue] = useState("42");
  const [error, setError] = useState("");

  const clearOutput = () => { setPrediction(null); setError(""); };
  const createImage = async () => {
    clearOutput(); setImage(null); setUploaded(null);
    try { const output = await onGenerate(className, Number(seedValue)); setImage(output); }
    catch (failure) { setError(failure.message); }
  };
  const classifyGenerated = async () => {
    if (!image) return;
    clearOutput();
    try { setPrediction(await onClassifyGenerated(image.image_id, arm, seed)); }
    catch (failure) { setError(failure.message); }
  };
  const classifyFile = async (file) => {
    clearOutput(); setImage(null); setUploaded(URL.createObjectURL(file));
    try { setPrediction(await onClassifyUpload(file, arm, seed)); }
    catch (failure) { setError(failure.message); }
  };
  const useGalleryImage = async (item) => {
    clearOutput(); setImage({ image_id: item.image_id, class_name: item.class_name, image_url: item.image_url, source: "previously generated sample" }); setUploaded(null);
  };
  const generatorReady = models?.generator?.available;
  const selectedClassifier = models?.classifiers?.find((item) => item.arm === arm && item.seed === seed);

  return (
    <main className="page-shell">
      <section className="page-intro"><div><p className="eyebrow">LIVE DEMONSTRATION · LOCAL INFERENCE</p><h1>Generate. Inspect. Classify.</h1><p>Use the existing LoRA adapter and one of the trained ResNet-18 runs. Inference is serialized to protect the GPU.</p></div><div className={`readiness ${generatorReady && selectedClassifier?.available ? "ready" : "not-ready"}`}><span className="status-dot" />{generatorReady ? "Local models available" : "Model files unavailable"}</div></section>

      <section className="demo-layout">
        <div className="panel demo-controls">
          <div className="panel-title"><span className="panel-symbol"><Icon>✳</Icon></span><div><p className="eyebrow">01 · GENERATE</p><h2>Synthetic defect</h2></div></div>
          <label className="field-label" htmlFor="defect-class">Defect class</label>
          <div className="class-picker" id="defect-class">
            {CLASSES.map(([key, title]) => <button key={key} onClick={() => {setClassName(key);clearOutput();}} className={className === key ? "class-choice chosen" : "class-choice"}><span className={`class-dot dot-${key}`} /><span>{title}</span>{className === key && <b>✓</b>}</button>)}
          </div>
          <label className="field-label" htmlFor="seed">Seed</label>
          <input id="seed" className="text-input" type="number" min="0" max="4294967295" value={seedValue} onChange={(event) => setSeedValue(event.target.value)} />
          <button className="button button-primary full-button" disabled={busy || !generatorReady} onClick={createImage}>{busy ? <><span className="spinner"/> Preparing GPU…</> : <><Icon>✦</Icon> Generate with SD 1.5 + LoRA</>}</button>
          <p className="micro-copy">Pinned local base revision · LoRA step 1,000 · generated demo images are stored outside the candidate dataset.</p>
          <div className="divider" />
          <div className="panel-title compact"><span className="panel-symbol panel-symbol-alt"><Icon>⌁</Icon></span><div><p className="eyebrow">02 · CLASSIFY</p><h2>ResNet-18 model</h2></div></div>
          <div className="field-row"><label className="field-label" htmlFor="arm">Training arm</label><label className="field-label" htmlFor="model-seed">Seed</label></div>
          <div className="field-row"><select id="arm" className="select-input" value={arm} onChange={(event) => {setArm(event.target.value);clearOutput();}}>{ARMS.map(([key,label]) => <option key={key} value={key}>{label}</option>)}</select><select id="model-seed" className="select-input" value={seed} onChange={(event) => {setSeed(Number(event.target.value));clearOutput();}}>{[42,43,44].map((value) => <option key={value} value={value}>Seed {value}</option>)}</select></div>
          <p className="model-caption">{selectedClassifier?.available ? `Verified checkpoint · ${selectedClassifier.run_id}` : "Selected checkpoint is not available."}</p>
        </div>

        <div className="panel demo-stage">
          <div className="stage-toolbar"><div><span className="live-dot"/> LOCAL INFERENCE STATION</div><span>{health?.device || "checking device"}</span></div>
          <div className="image-stage">
            {busy && <div className="stage-loading"><span className="spinner spinner-large"/><strong>{health?.active_task || "Processing image"}</strong><p>GPU work is queued and running off the UI thread. Keep this page open.</p></div>}
            {!busy && image && <div className="result-image-wrap"><img className="result-image" src={image.image_url} alt={`${classLabel(image.class_name)} synthetic steel defect`} /><div className="image-caption"><span>{classLabel(image.class_name)}</span><span>Seed {image.seed ?? seedValue}</span><span>{image.source || "New live sample · pending review"}</span></div></div>}
            {!busy && !image && !uploaded && <div className="stage-empty"><div className="target-frame"><span/><span/><span/><span/><div className="empty-grain">···<br/>····<br/>· · ·</div></div><p>IMAGE OUTPUT</p><small>Choose a defect class, then generate or upload an image.</small></div>}
            {!busy && uploaded && <div className="result-image-wrap"><img className="result-image" src={uploaded} alt="Uploaded image being classified" /><div className="image-caption"><span>Uploaded sample</span><span>Not added to training data</span></div></div>}
          </div>
          {image && !prediction && <div className="classify-action"><p>Image ready. Run the selected checkpoint to see its actual prediction.</p><button className="button button-primary" disabled={busy} onClick={classifyGenerated}>Classify with {armLabel(arm)} · {seed} <span>→</span></button></div>}
          {prediction && <PredictionCard result={prediction} />}
          {error && <div className="notice notice-error"><span>!</span><div><strong>Could not complete this request</strong><p>{error}</p></div><button onClick={() => setError("")} aria-label="Dismiss">×</button></div>}
        </div>
      </section>

      <section className="content-section gallery-section">
        <SectionHeading kicker="DEMONSTRATION GALLERY" title="Previously generated examples">These are deterministic samples from the visual-review contact sheets. They are not acceptance decisions.</SectionHeading>
        {gallery?.length ? <div className="gallery-grid">{gallery.map((item) => <button className="gallery-item" key={item.image_id} onClick={() => useGalleryImage(item)}><img src={item.image_url} loading="lazy" alt={`Exploratory ${classLabel(item.class_name)} sample`} /><span><strong>{classLabel(item.class_name)}</strong><small>Review sample · pending</small></span></button>)}</div> : <div className="empty-card">Gallery samples are unavailable. The live generator remains available when local models are ready.</div>}
        <p className="micro-copy">Selecting a gallery example loads it into the viewer; it does not accept or reject that candidate.</p>
      </section>
    </main>
  );
}

function PredictionCard({ result }) {
  return <div className="prediction-card"><div className="prediction-top"><div><p className="eyebrow">MODEL PREDICTION</p><h3>{classLabel(result.predicted_class)}</h3><p>{result.model_name}</p></div><div className="confidence"><strong>{pct(result.confidence)}</strong><small>confidence</small></div></div><div className="probability-list">{result.probabilities.map((item) => <div className="probability-row" key={item.class_name}><span>{classLabel(item.class_name)}</span><div className="probability-track"><i style={{ width: `${Math.max(item.score * 100, 1)}%` }} /></div><b>{pct(item.score)}</b></div>)}</div><p className="micro-copy">Prediction is the selected checkpoint’s output; it is not a human defect diagnosis.</p></div>;
}

function ResultsPage({ results, loading, error, goOverview }) {
  const [selectedArm, setSelectedArm] = useState("real_only");
  const [figureModal, setFigureModal] = useState(null);
  const perClassF1 = useMemo(() => {
    const rows = results?.class_aggregates?.filter((row) => row.metric === "f1") ?? [];
    return CLASSES.map(([className, label]) => ({
      className, label,
      ...Object.fromEntries(ARMS.map(([arm]) => [arm, rows.find((row) => row.class_name === className && row.arm === arm)])),
    }));
  }, [results]);
  if (loading) return <Loading label="Reading verified Phase 1 results…" />;
  if (error || !results) return <ErrorPanel message={error || "Evaluation results are unavailable."} onBack={goOverview} />;

  const armRuns = results.runs.filter((run) => run.arm === selectedArm).sort((a, b) => a.seed - b.seed);
  return (
    <main className="page-shell">
      <section className="page-intro results-intro"><div><p className="eyebrow">PHASE 1 · FROZEN TEST EVALUATION</p><h1>What did the experiments show?</h1><p>Nine saved checkpoints · one shared set of 270 real images · three matched seeds.</p></div><div className="verified-tag"><span>✓</span> Hash-verified evaluation</div></section>
      <div className="notice notice-blue"><span>i</span><div><strong>Read the result with its ceiling in mind</strong><p>Real-only reached 100% accuracy and macro F1 on this frozen split. The mixed arm did not improve on that score. This ceiling makes the measured synthetic-augmentation effect non-positive here; it does not establish that augmentation can never help.</p></div></div>

      <section className="content-section">
        <SectionHeading kicker="MEAN ± SAMPLE SD · N=3 SEEDS" title="Arm-level test performance">Macro metrics give each of the six classes equal weight. Values come directly from the Phase 1 evaluation JSON.</SectionHeading>
        <div className="results-table-wrap"><table className="results-table"><thead><tr><th>Training arm</th>{METRICS.map(([,name]) => <th key={name}>{name}</th>)}</tr></thead><tbody>{ARMS.map(([arm,label]) => <tr key={arm}><td><span className={`arm-bullet arm-bullet-${arm}`} />{label}</td>{METRICS.map(([key]) => {const cell=results.aggregates[arm][key]; return <td key={key}><strong>{pct(cell.mean)}</strong><small>± {pct(cell.sample_sd)} · n={cell.n}</small></td>;})}</tr>)}</tbody></table></div>
      </section>

      <section className="content-section">
        <SectionHeading kicker="MEASURED FIGURES" title="Accuracy, F1 and class-level behavior" />
        <div className="figure-grid">{FIGURES.map(([filename,title]) => <FigureCard key={filename} filename={filename} title={title} onOpen={() => setFigureModal([filename,title])} />)}</div>
      </section>

      <section className="content-section">
        <SectionHeading kicker="ERROR ANALYSIS" title="Confusion matrices by training arm">Each pooled matrix combines the three seeds: 810 predictions in total. Rows show the true class; columns show the model prediction.</SectionHeading>
        <div className="figure-grid confusion-grid">{CONFUSIONS.map(([filename,title]) => <FigureCard key={filename} filename={filename} title={title} onOpen={() => setFigureModal([filename,title])} />)}</div>
      </section>

      <section className="two-column results-detail">
        <article className="panel panel-padding">
          <div className="detail-heading"><div><p className="eyebrow">CLASS-WISE F1</p><h2>Which defects transferred?</h2></div><span className="tiny-badge">mean ± SD</span></div>
          <div className="class-table-wrap"><table className="class-table"><thead><tr><th>Class</th>{ARMS.map(([arm])=><th key={arm}>{armLabel(arm)}</th>)}</tr></thead><tbody>{perClassF1.map((row)=><tr key={row.className}><td>{row.label}</td>{ARMS.map(([arm])=><td key={arm}><strong>{pct(Number(row[arm]?.mean))}</strong><small>± {pct(Number(row[arm]?.sample_sd))}</small></td>)}</tr>)}</tbody></table></div>
        </article>
        <article className="panel panel-padding seed-panel">
          <div className="detail-heading"><div><p className="eyebrow">INDIVIDUAL RUNS</p><h2>Seed-wise scorecard</h2></div><select className="select-input arm-filter" value={selectedArm} onChange={(event)=>setSelectedArm(event.target.value)}>{ARMS.map(([arm,label])=><option value={arm} key={arm}>{label}</option>)}</select></div>
          <div className="seed-cards">{armRuns.map((run)=><div className="seed-run" key={`${run.arm}-${run.seed}`}><div className="seed-run-title"><span>SEED {run.seed}</span><small>{run.run_id}</small></div><div className="seed-score-row">{METRICS.map(([key,label])=><div key={key}><small>{label.replace("Macro ","M-")}</small><strong>{pct(run.metrics[key])}</strong></div>)}</div></div>)}</div>
        </article>
      </section>

      <section className="content-section provenance-strip"><div><p className="eyebrow">EVALUATION PROVENANCE</p><p>Frozen manifest SHA-256</p><code>{results.manifest_sha256}</code></div><div><p className="eyebrow">SHARED TEST MEMBERSHIP</p><p>All nine runs used the same IDs</p><code>{results.test_membership_sha256}</code></div><div><p className="eyebrow">SOFTWARE</p><p>{results.device} · PyTorch {results.torch}</p><small>Final evaluation only · no test-set model selection</small></div></section>

      {figureModal && <div className="modal-backdrop" role="presentation" onClick={()=>setFigureModal(null)}><div className="figure-modal" role="dialog" aria-modal="true" aria-label={figureModal[1]} onClick={(event)=>event.stopPropagation()}><div><strong>{figureModal[1]}</strong><button onClick={()=>setFigureModal(null)} aria-label="Close figure">×</button></div><img src={`/api/figures/${figureModal[0]}`} alt={figureModal[1]} /><a href={`/api/figures/${figureModal[0].replace(".png",".pdf")}`} target="_blank" rel="noreferrer">Open print-ready PDF ↗</a></div></div>}
    </main>
  );
}

function FigureCard({ filename, title, onOpen }) {
  return <button className="figure-card" onClick={onOpen}><div className="figure-preview"><img src={`/api/figures/${filename}`} alt={title} loading="lazy" /></div><span><strong>{title}</strong><small>Open figure ↗</small></span></button>;
}

function Loading({ label }) { return <main className="page-shell"><div className="loading-panel"><span className="spinner spinner-large"/><strong>{label}</strong><p>Reading local project files. No model is loading.</p></div></main>; }
function ErrorPanel({ message, onBack }) { return <main className="page-shell"><div className="error-page"><span>!</span><h2>Could not load this section</h2><p>{message}</p>{onBack && <button className="button button-primary" onClick={onBack}>Back to overview</button>}</div></main>; }

function Footer() {
  return <footer className="footer"><div><span className="footer-mark">S</span><span>STEELSIGHT <small>RESEARCH PRESENTATION</small></span></div><p>Local-only demonstration · Research use · Not an industrial inspection system</p><span>NEU-CLS · 2026</span></footer>;
}

export default function App() {
  const [page, setPage] = useState("overview");
  const [health, setHealth] = useState(null);
  const [overview, setOverview] = useState(null);
  const [results, setResults] = useState(null);
  const [models, setModels] = useState(null);
  const [gallery, setGallery] = useState([]);
  const [loading, setLoading] = useState(true);
  const [resultsError, setResultsError] = useState("");
  const [busy, setBusy] = useState(false);

  const refreshHealth = useCallback(async () => {
    try { setHealth(await request("/api/health")); } catch { setHealth({ status: "offline", cuda_available: false }); }
  }, []);
  useEffect(() => {
    let active = true;
    Promise.all([
      request("/api/health"), request("/api/project/overview"), request("/api/research/results"),
      request("/api/models"), request("/api/gallery?limit_per_class=3"),
    ]).then(([status, project, evaluation, available, samples]) => {
      if (!active) return;
      setHealth(status); setOverview(project); setResults(evaluation); setModels(available); setGallery(samples.items ?? []);
    }).catch((error) => { if (active) setResultsError(error.message); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);
  useEffect(() => {
    if (!busy) return undefined;
    const timer = window.setInterval(refreshHealth, 1000);
    return () => window.clearInterval(timer);
  }, [busy, refreshHealth]);

  const doRequest = async (callback) => {
    setBusy(true);
    try { return await callback(); }
    finally { setBusy(false); await refreshHealth(); }
  };
  const generate = (class_name, seed) => doRequest(() => request("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ class_name, seed }) }));
  const classifyGenerated = (id, arm, seed) => doRequest(() => request(`/api/classify-generated/${encodeURIComponent(id)}?arm=${encodeURIComponent(arm)}&seed=${seed}`, { method: "POST" }));
  const classifyUpload = (file, arm, seed) => doRequest(() => { const form = new FormData(); form.append("file", file); return request(`/api/classify?arm=${encodeURIComponent(arm)}&seed=${seed}`, { method: "POST", body: form }); });

  return (
    <div className="app-frame">
      <Header page={page} setPage={setPage} health={health} />
      {resultsError && !results && <div className="global-error">Backend/data issue: {resultsError}</div>}
      {page === "overview" && <OverviewPage overview={overview} results={results} goDemo={() => setPage("demo")} goResults={() => setPage("results")} loading={loading} />}
      {page === "demo" && <DemoPage health={health} models={models} gallery={gallery} onGenerate={generate} onClassifyUpload={classifyUpload} onClassifyGenerated={classifyGenerated} busy={busy} />}
      {page === "results" && <ResultsPage results={results} loading={loading} error={resultsError} goOverview={() => setPage("overview")} />}
      <Footer />
    </div>
  );
}
