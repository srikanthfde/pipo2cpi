// PIPO2CPI frontend: UI only. Served by main.py and compiled in the browser (React 18 + Babel).
const { useState, useEffect, useRef } = React;

const CSS = `
:root{--bg:#0b0d10;--panel:#13171c;--panel2:#191e25;--line:#262c35;--text:#e8e6e1;--muted:#9aa0a8;
--accent:#f0632e;--accent-bg:#2a1912;--ok:#4cc38a;--ok-bg:#12261d;--warn:#f2b34a;--warn-bg:#2a2112;--err:#f26b6b;--err-bg:#2a1414;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
button{font:inherit;cursor:pointer;border-radius:6px;border:1px solid var(--line);background:var(--panel2);color:var(--text);padding:7px 14px}
button:hover:not(:disabled){border-color:var(--accent)}button:disabled{opacity:.45;cursor:not-allowed}
button.primary{background:var(--accent);border-color:var(--accent);color:#140a05;font-weight:600}
button.ghost{background:transparent}
input,textarea,select{width:100%;font:13px var(--mono);background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:7px 9px}
textarea{resize:vertical;min-height:64px}label{display:block;font-size:12px;color:var(--muted);margin:10px 0 4px}
.top{display:flex;align-items:center;gap:16px;padding:14px 28px;border-bottom:1px solid var(--line)}
.brand{font:700 22px var(--mono);letter-spacing:.5px}.brand b{color:var(--accent)}
.sub{color:var(--muted);font-size:13px}.spacer{flex:1}
.pill{font:12px var(--mono);padding:3px 9px;border-radius:99px;border:1px solid var(--line);color:var(--muted)}
.pill.ok{color:var(--ok);border-color:var(--ok)}.pill.err{color:var(--err);border-color:var(--err)}
.tabs{display:flex;gap:4px;padding:0 28px;border-bottom:1px solid var(--line)}
.tab{background:none;border:none;border-bottom:2px solid transparent;border-radius:0;padding:12px 14px;color:var(--muted)}
.tab.on{color:var(--text);border-bottom-color:var(--accent)}
.wrap{padding:22px 28px;max-width:1500px;margin:0 auto}
.grid{display:grid;grid-template-columns:360px 1fr;gap:22px;align-items:start}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.panel h3{margin:0 0 4px;font-size:15px}.eyebrow{font:12px var(--mono);color:var(--accent);letter-spacing:1px;text-transform:uppercase}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
.stage{background:var(--panel);border:1px solid var(--line);border-top:2px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:12px}
.stage.running{border-top-color:var(--warn)}.stage.done{border-top-color:var(--accent)}.stage.error{border-top-color:var(--err)}
.shead{display:flex;align-items:center;gap:10px}.num{font:12px var(--mono);color:var(--accent)}
.stitle{font-weight:600;font-size:15px}.type{font:11px var(--mono);color:var(--muted);border:1px solid var(--line);padding:2px 7px;border-radius:4px}
.status{margin-left:auto;font:12px var(--mono)}.desc{color:var(--muted);font-size:13px;margin:4px 0 0}
.gate{background:var(--warn-bg);border:1px solid var(--warn);border-radius:10px;padding:12px 16px;margin:-4px 0 12px}
.gate b{color:var(--warn)}.gate p{margin:4px 0 10px;color:var(--text)}
.err{background:var(--err-bg);border:1px solid var(--err);color:var(--err);border-radius:8px;padding:9px 12px;margin-top:10px;font-size:13px}
.warn{background:var(--warn-bg);color:var(--warn);border-radius:6px;padding:6px 10px;margin-top:8px;font-size:12px}
.out{margin-top:12px;border-top:1px solid var(--line);padding-top:10px}
.kv{display:grid;grid-template-columns:170px 1fr;gap:6px 14px}.k{color:var(--muted);font:12px var(--mono);padding-top:2px}
table{width:100%;border-collapse:collapse;font-size:12.5px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:500;font-family:var(--mono);font-size:11.5px}
pre{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;overflow:auto;font:12px var(--mono);white-space:pre-wrap;margin:0}
ul{margin:0;padding-left:18px}.chip{display:inline-block;font:11.5px var(--mono);padding:2px 8px;margin:6px 6px 0 0;border-radius:4px;background:var(--panel2);color:var(--muted)}
.chip.cited{color:var(--accent);background:var(--accent-bg)}
details{margin-top:8px}summary{cursor:pointer;color:var(--muted);font-size:12.5px}
.badge{font:11.5px var(--mono);padding:1px 7px;border-radius:4px}.badge.t{background:var(--ok-bg);color:var(--ok)}.badge.f{background:var(--err-bg);color:var(--err)}
.chat{display:flex;flex-direction:column;gap:10px;min-height:340px}.msg{max-width:80%;padding:10px 14px;border-radius:10px;white-space:pre-wrap}
.msg.user{align-self:flex-end;background:var(--accent-bg);border:1px solid #4a2616}.msg.assistant{align-self:flex-start;background:var(--panel2);border:1px solid var(--line)}
.foot{color:var(--muted);font:12px var(--mono);padding:16px 28px;border-top:1px solid var(--line);margin-top:24px}
.spin{display:inline-block;width:10px;height:10px;border:2px solid var(--warn);border-right-color:transparent;border-radius:50%;animation:s .8s linear infinite}
@keyframes s{to{transform:rotate(360deg)}}
.scen{margin-bottom:22px}.scen-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.scen-list{max-height:320px;overflow:auto;margin-top:12px;border:1px solid var(--line);border-radius:8px}
.scen-list tr{cursor:pointer}.scen-list tr:hover td{background:var(--panel2)}.scen-list tr.on td{background:var(--accent-bg)}
.scen-list td{font-family:var(--mono);font-size:12px}
@media(max-width:1000px){.grid{grid-template-columns:1fr}}
`;

const EXAMPLE_ICO = {
  po_system: "Q45 (7.50)",
  sender_party: "",
  sender_component: "BS_Q96_CL100",
  sender_interface: "MaterialERPAvailabilityQueryResponse_Out",
  namespace: "urn:tooling.sandvik.com:MarketingBusiness:GlobalDataTypes",
  receiver_component: "",
  naming_prefix: "",
  po_export: "Integrated Configuration, synchronous (Best Effort). Sender channel SOAP. Receiver channel SOAP.",
  mapping_code: "",
  sample_input: "",
  expected_output: "",
  actual_output: "",
  mpl_error: "",
};

const FIELD_META = {
  po_system: ["PO system", "input"], sender_party: ["Sender party", "input"],
  sender_component: ["Sender component", "input"], sender_interface: ["Sender interface", "input"],
  namespace: ["Namespace", "input"], receiver_component: ["Receiver component", "input"],
  naming_prefix: ["iFlow ID prefix (optional)", "input"],
  po_export: ["ICo / channel export text", "area"], mapping_code: ["Java mapping / UDF code", "area"],
  sample_input: ["Sample input payload", "area"], expected_output: ["Expected output (PO)", "area"],
  actual_output: ["Actual output (CPI)", "area"], mpl_error: ["MPL error text", "area"],
};

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}

const pretty = (k) => k.replace(/_/g, " ");

function Value({ v, k }) {
  if (v === null || v === undefined || v === "") return <span className="sub">—</span>;
  if (typeof v === "boolean") return <span className={"badge " + (v ? "t" : "f")}>{String(v)}</span>;
  if (typeof v === "string" || typeof v === "number") {
    const s = String(v);
    if ((k && /code|spec/.test(k)) || s.includes("\n")) return <pre>{s}</pre>;
    return <span>{s}</span>;
  }
  if (Array.isArray(v)) {
    if (!v.length) return <span className="sub">none</span>;
    if (v.every((x) => x && typeof x === "object" && !Array.isArray(x))) {
      const cols = [...new Set(v.flatMap((x) => Object.keys(x)))];
      return (
        <table><thead><tr>{cols.map((c) => <th key={c}>{pretty(c)}</th>)}</tr></thead>
          <tbody>{v.map((row, i) => <tr key={i}>{cols.map((c) => <td key={c}><Value v={row[c]} k={c} /></td>)}</tr>)}</tbody>
        </table>
      );
    }
    return <ul>{v.map((x, i) => <li key={i}><Value v={x} /></li>)}</ul>;
  }
  return (
    <div className="kv">{Object.entries(v).map(([kk, vv]) => (
      <React.Fragment key={kk}><div className="k">{pretty(kk)}</div><div><Value v={vv} k={kk} /></div></React.Fragment>
    ))}</div>
  );
}

function StageCard({ index, stage, result }) {
  const status = result?.status || "idle";
  const label = { idle: "waiting", running: "running", done: "done", error: "error", rejected: "stopped" }[status];
  const r = result?.data;
  return (
    <div className={"stage " + status}>
      <div className="shead">
        <span className="num">{String(index + 1).padStart(2, "0")}</span>
        <span className="stitle">{stage.title}</span>
        <span className="type">{stage.agent_type}</span>
        <span className="status">{status === "running" && <span className="spin" />} {label}</span>
      </div>
      <p className="desc">{stage.description}</p>
      {result?.error && <div className="err">{result.error}</div>}
      {r && (
        <div className="out">
          <Value v={r.output} />
          {r.warnings?.map((w, i) => <div className="warn" key={i}>{w}</div>)}
          <div>{r.citations?.map((c) => (
            <span key={c.id} className={"chip" + (c.cited ? " cited" : "")} title={`score ${c.score}`}>kb:{c.id}</span>
          ))}</div>
          <details><summary>Agent loop trace · {r.trace.length} tool call(s){r.repairs ? ` · ${r.repairs} repair` : ""}</summary>
            {r.trace.length ? <Value v={r.trace} /> : <p className="sub">Answered without tools.</p>}
          </details>
        </div>
      )}
    </div>
  );
}

function ScenarioBrowser({ meta, onPick }) {
  const system = meta.pipo?.system || "S45";
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [picked, setPicked] = useState("");

  async function load() {
    setBusy(true); setError("");
    try { setData(await api("/api/scenarios")); } catch (e) { setError(e.message); setData(null); }
    setBusy(false);
  }

  const q = query.trim().toLowerCase();
  const rows = (data?.scenarios || []).filter((r) => !q || r.scenario_string.toLowerCase().includes(q));

  function pick(r) {
    setPicked(r.scenario_string);
    onPick({
      po_system: system, sender_party: r.sender_party, sender_component: r.sender_component,
      sender_interface: r.interface, namespace: r.namespace, receiver_component: r.receiver_component,
    });
  }

  return (
    <div className="panel scen">
      <div className="scen-head">
        <div>
          <div className="eyebrow">SAP PI/PO {system}</div>
          <h3 style={{ margin: 0 }}>Integration Directory scenarios</h3>
        </div>
        <button className="primary" disabled={busy} onClick={load}>{busy ? "Fetching..." : "Get scenarios"}</button>
        {data && <span className="pill ok">{data.count} scenarios · {new Date(data.fetched_at).toLocaleTimeString()}</span>}
        {data && <input style={{ flex: 1, minWidth: 220 }} value={query} placeholder="Filter by component, interface or namespace..."
          onChange={(e) => setQuery(e.target.value)} />}
      </div>
      {!meta.pipo?.configured && !data && !error && (
        <p className="sub" style={{ margin: "10px 0 0" }}>{system} is only reachable when PIPO2CPI runs on your PC on the company network or VPN, with the {system} settings in .env.</p>
      )}
      {error && <div className="err">{error}</div>}
      {data && (
        <div className="scen-list">
          <table>
            <thead><tr><th>Sender component</th><th>Interface</th><th>Namespace</th><th>Receiver</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.scenario_string} className={picked === r.scenario_string ? "on" : ""} onClick={() => pick(r)}
                  title="Use this scenario in the pipeline">
                  <td>{r.sender_party ? `${r.sender_party} | ` : ""}{r.sender_component}</td>
                  <td>{r.interface}</td><td>{r.namespace}</td>
                  <td>{[r.receiver_party, r.receiver_component].filter(Boolean).join(" | ") || "—"}</td>
                </tr>
              ))}
              {!rows.length && <tr><td colSpan={4} className="sub">No scenarios match "{query}".</td></tr>}
            </tbody>
          </table>
        </div>
      )}
      {data && <p className="sub" style={{ margin: "8px 0 0", fontSize: 12 }}>Click a scenario to load it into the pipeline form below.</p>}
    </div>
  );
}

function Pipeline({ meta }) {
  const stages = meta.pipeline;
  const [ico, setIco] = useState(EXAMPLE_ICO);
  const [docs, setDocs] = useState("");
  const [results, setResults] = useState({});
  const [gateAt, setGateAt] = useState(null);
  const [running, setRunning] = useState(false);
  const [showMore, setShowMore] = useState(false);

  const setRes = (key, val) => setResults((p) => ({ ...p, [key]: val }));

  async function runFrom(start, known) {
    setRunning(true);
    setGateAt(null);
    const outputs = { ...known };
    const documents = docs.split(/\n-{3,}\n/).map((d) => d.trim()).filter(Boolean);
    for (let i = start; i < stages.length; i++) {
      const s = stages[i];
      setRes(s.key, { status: "running" });
      try {
        const data = await api("/api/stage", { stage: s.key, ico, prior: outputs, documents });
        outputs[s.key] = data.output;
        setRes(s.key, { status: "done", data });
        if (s.gate) { setGateAt(i); break; }
      } catch (e) {
        setRes(s.key, { status: "error", error: e.message });
        break;
      }
    }
    setRunning(false);
  }

  const knownOutputs = () =>
    Object.fromEntries(Object.entries(results).filter(([, v]) => v.status === "done").map(([k, v]) => [k, v.data.output]));

  function start() { setResults({}); runFrom(0, {}); }
  function approve() { const i = gateAt; setGateAt(null); runFrom(i + 1, knownOutputs()); }
  function reject() { const s = stages[gateAt]; setGateAt(null); setRes(s.key, { ...results[s.key], status: "rejected" }); }
  function retry(i) { runFrom(i, knownOutputs()); }

  const failedAt = stages.findIndex((s) => ["error", "rejected"].includes(results[s.key]?.status));
  const mainFields = ["po_system", "sender_component", "sender_interface", "namespace", "receiver_component", "po_export"];
  const moreFields = Object.keys(EXAMPLE_ICO).filter((f) => !mainFields.includes(f));

  const field = (f) => {
    const [lbl, kind] = FIELD_META[f];
    const on = (e) => setIco({ ...ico, [f]: e.target.value });
    return (
      <div key={f}><label>{lbl}</label>
        {kind === "area" ? <textarea value={ico[f]} onChange={on} /> : <input value={ico[f]} onChange={on} />}
      </div>
    );
  };

  return (
    <>
    <ScenarioBrowser meta={meta} onPick={(f) => {
      if (running) return;
      setIco((prev) => ({ ...prev, po_export: "", mapping_code: "", sample_input: "", expected_output: "",
                          actual_output: "", mpl_error: "", ...f }));
      setResults({}); setGateAt(null);
    }} />
    <div className="grid">
      <div className="panel">
        <div className="eyebrow">Integrated Configuration</div>
        <h3>ICo to migrate</h3>
        {mainFields.map(field)}
        <details open={showMore} onToggle={(e) => setShowMore(e.target.open)}>
          <summary>Mappings, payloads and MPL errors</summary>
          {moreFields.map(field)}
        </details>
        <label>Reference documents for RAG (optional, separate with a line of ---)</label>
        <textarea value={docs} onChange={(e) => setDocs(e.target.value)} placeholder="Paste team guidelines, channel exports, specs..." />
        <div className="row">
          <button className="primary" disabled={running || !meta.configured} onClick={start}>Run pipeline</button>
          <button className="ghost" disabled={running} onClick={() => { setResults({}); setGateAt(null); }}>Reset</button>
          <button className="ghost" disabled={running} onClick={() => setIco(EXAMPLE_ICO)}>Load example</button>
        </div>
        {!meta.configured && <div className="warn">Set OPENROUTER_API_KEY in .env and restart to run agents.</div>}
      </div>
      <div>
        {stages.map((s, i) => (
          <React.Fragment key={s.key}>
            <StageCard index={i} stage={s} result={results[s.key]} />
            {gateAt === i && (
              <div className="gate">
                <b>Human checkpoint: {s.gate.title}</b>
                <p>{s.gate.instruction}</p>
                <div className="row" style={{ marginTop: 0 }}>
                  <button className="primary" onClick={approve}>Approve and continue</button>
                  <button onClick={reject}>Reject</button>
                </div>
              </div>
            )}
            {failedAt === i && !running && (
              <div className="row" style={{ margin: "-4px 0 12px" }}><button onClick={() => retry(i)}>Re-run this stage</button></div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
    </>
  );
}

function Chat({ meta }) {
  const [agentKey, setAgentKey] = useState("orchestrator");
  const [msgs, setMsgs] = useState([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const end = useRef(null);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs, busy]);

  async function send() {
    const message = text.trim();
    if (!message) return;
    setText(""); setError(""); setBusy(true);
    const history = msgs.map(({ role, content }) => ({ role, content }));
    setMsgs((m) => [...m, { role: "user", content: message }]);
    try {
      const r = await api("/api/chat", { agent: agentKey, message, history });
      setMsgs((m) => [...m, { role: "assistant", content: r.answer, citations: r.citations, warnings: r.warnings }]);
    } catch (e) {
      setError(e.message);
      setMsgs((m) => m.slice(0, -1));
    }
    setBusy(false);
  }

  return (
    <div className="panel">
      <div className="row" style={{ marginTop: 0, alignItems: "center" }}>
        <span className="sub">Ask</span>
        <select style={{ width: 280 }} value={agentKey} onChange={(e) => setAgentKey(e.target.value)}>
          {meta.chat_agents.map((a) => <option key={a.key} value={a.key}>{a.title}</option>)}
        </select>
        <button className="ghost" onClick={() => setMsgs([])}>Clear</button>
      </div>
      <div className="chat" style={{ marginTop: 14 }}>
        {!msgs.length && <p className="sub">Example: "Can I migrate a sender RFC channel?" or "What do I check after clicking Migrate?"</p>}
        {msgs.map((m, i) => (
          <div key={i} className={"msg " + m.role}>
            {m.content}
            {m.warnings?.map((w, j) => <div className="warn" key={j}>{w}</div>)}
            {m.citations && <div>{m.citations.map((c) => <span className="chip" key={c.id}>kb:{c.id}</span>)}</div>}
          </div>
        ))}
        {busy && <div className="msg assistant"><span className="spin" /> thinking</div>}
        {error && <div className="err">{error}</div>}
        <div ref={end} />
      </div>
      <div className="row">
        <input style={{ flex: 1 }} value={text} placeholder="Ask about PI/PO to CPI migration..."
          onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && !busy && send()} />
        <button className="primary" disabled={busy || !meta.configured} onClick={send}>Send</button>
      </div>
    </div>
  );
}

function Evals({ meta }) {
  const [includeLlm, setIncludeLlm] = useState(false);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function run() {
    setBusy(true); setError("");
    try { setRes(await api("/api/eval", { include_llm: includeLlm })); } catch (e) { setError(e.message); }
    setBusy(false);
  }

  return (
    <div className="panel">
      <div className="eyebrow">Evaluation engineering</div>
      <h3>Test the harness</h3>
      <p className="sub">Deterministic cases check guardrails, tools, retrieval, context building and the loop's validators.
        Live cases call the model with the team's golden ICo and assert on the result.</p>
      <div className="row" style={{ alignItems: "center" }}>
        <label style={{ display: "flex", gap: 8, alignItems: "center", margin: 0, color: "var(--text)" }}>
          <input type="checkbox" style={{ width: "auto" }} checked={includeLlm} disabled={!meta.configured}
            onChange={(e) => setIncludeLlm(e.target.checked)} /> Include live model cases
        </label>
        <button className="primary" disabled={busy} onClick={run}>{busy ? "Running..." : "Run evals"}</button>
        {res && <span className={"pill " + (res.failed ? "err" : "ok")}>{res.passed}/{res.total} passed</span>}
      </div>
      {error && <div className="err">{error}</div>}
      {res && (
        <table style={{ marginTop: 14 }}>
          <thead><tr><th>Result</th><th>Case</th><th>Category</th><th>Detail</th></tr></thead>
          <tbody>{res.results.map((r) => (
            <tr key={r.name}>
              <td><span className={"badge " + (r.passed ? "t" : "f")}>{r.passed ? "PASS" : "FAIL"}</span></td>
              <td>{r.name}</td><td className="sub">{r.category}</td>
              <td style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--muted)" }}>{r.detail}</td>
            </tr>
          ))}</tbody>
        </table>
      )}
    </div>
  );
}

function App() {
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("pipeline");
  useEffect(() => { api("/api/agents").then(setMeta).catch((e) => setError(e.message)); }, []);

  return (
    <>
      <style>{CSS}</style>
      <header className="top">
        <div className="brand">PIPO<b>2</b>CPI</div>
        <div className="sub">SAP PI/PO to Cloud Integration migration agents</div>
        <div className="spacer" />
        {meta && <span className={"pill " + (meta.configured ? "ok" : "err")}>
          {meta.configured ? meta.model : "API key missing"}</span>}
      </header>
      <nav className="tabs">
        {[["pipeline", "Migration pipeline"], ["chat", "Ask an agent"], ["evals", "Evaluations"]].map(([k, l]) => (
          <button key={k} className={"tab" + (tab === k ? " on" : "")} onClick={() => setTab(k)}>{l}</button>
        ))}
      </nav>
      <main className="wrap">
        {error && <div className="err">Could not reach the API: {error}</div>}
        {!meta && !error && <p className="sub"><span className="spin" /> Loading agents...</p>}
        {meta && tab === "pipeline" && <Pipeline meta={meta} />}
        {meta && tab === "chat" && <Chat meta={meta} />}
        {meta && tab === "evals" && <Evals meta={meta} />}
      </main>
      <footer className="foot">Agents recommend. People click Migrate, approve deploys and sign off cutover.</footer>
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
