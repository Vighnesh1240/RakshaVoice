import { useEffect, useRef, useState } from "react";
import "./App.css";

/* ------------------------------------------------------------------ */
/*  BACKEND CONTRACT (unchanged)                                       */
/* ------------------------------------------------------------------ */
const API_URL = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws";

const LABELS = ["urgency", "authority", "otp_pii", "payment"];
const LABEL_NAMES = {
  urgency: "Urgency",
  authority: "Authority claim",
  otp_pii: "OTP / PIN / card request",
  payment: "Payment request",
};
const EVIDENCE_NAMES = {
  spoof: "Spoof signal",
  urgency: "Urgency",
  authority: "Authority claim",
  otp_pii: "OTP / PII request",
  payment: "Payment request",
  voice_mismatch: "Voice mismatch",
};

/* ------------------------------------------------------------------ */
/*  UI-only helpers                                                    */
/* ------------------------------------------------------------------ */
const clamp01 = (v) => Math.min(1, Math.max(0, typeof v === "number" && Number.isFinite(v) ? v : 0));
const pct = (v) => Math.round(clamp01(v) * 100);
const MAX_POINTS = 30;

function newStats() {
  return { windows: 0, peak: 0, alertWindows: 0, latSum: 0, latN: 0, signals: {}, startedAt: null };
}

function maxIntent(intent) {
  if (!intent?.labels) return null;
  const nums = Object.values(intent.labels).filter((v) => typeof v === "number");
  return nums.length ? Math.max(0, ...nums) : 0;
}

function riskTone(risk, threshold, alert) {
  if (alert || risk >= threshold) return "high";
  if (risk >= threshold * 0.55) return "mid";
  return "low";
}

function useAnimatedNumber(target, duration = 650) {
  const [val, setVal] = useState(target ?? 0);
  const fromRef = useRef(target ?? 0);
  useEffect(() => {
    const to = target ?? 0;
    const from = fromRef.current;
    const start = performance.now();
    let raf;
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const v = from + (to - from) * eased;
      fromRef.current = v;
      setVal(v);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return val;
}

function Icon({ name }) {
  const common = { width: 20, height: 20, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true };
  switch (name) {
    case "shield":
      return (<svg {...common}><path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.500-7-10V6l7-3z" /><path d="M9 12l2 2 4-4" /></svg>);
    case "wave":
      return (<svg {...common}><path d="M3 12h2M7 8v8M11 5v14M15 8v8M19 10v4M21 12h0" /></svg>);
    case "text":
      return (<svg {...common}><path d="M4 6h16M4 12h10M4 18h13" /></svg>);
    case "user":
      return (<svg {...common}><circle cx="12" cy="8" r="4" /><path d="M4 21c1.500-4 5-6 8-6s6.500 2 8 6" /></svg>);
    case "sun":
      return (<svg {...common}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.500 1.500M17.500 17.500L19 19M5 19l1.500-1.500M17.500 6.500L19 5" /></svg>);
    case "moon":
      return (<svg {...common}><path d="M20 14.500A8 8 0 019.500 4 8 8 0 1020 14.500z" /></svg>);
    case "lock":
      return (<svg {...common}><rect x="5" y="11" width="14" height="9" rx="2" /><path d="M8 11V8a4 4 0 018 0v3" /></svg>);
    case "upload":
      return (<svg {...common}><path d="M12 16V4M7 9l5-5 5 5M5 20h14" /></svg>);
    case "spark":
      return (<svg {...common}><path d="M12 3l1.800 5.200L19 10l-5.200 1.800L12 17l-1.800-5.200L5 10l5.200-1.800L12 3z" /></svg>);
    default:
      return null;
  }
}

/* ------------------------------------------------------------------ */
/*  Visual components (presentation only)                              */
/* ------------------------------------------------------------------ */
function RiskGauge({ value, threshold, tone, available }) {
  const animated = useAnimatedNumber(available ? clamp01(value) : 0);
  const angle = Math.PI * (1 - clamp01(threshold));
  const tx = (r) => 150 + r * Math.cos(angle);
  const ty = (r) => 150 - r * Math.sin(angle);
  return (
    <div className={`gauge tone-${tone}`}>
      <svg viewBox="0 0 300 170" role="img" aria-label={available ? `Combined model risk ${pct(value)} percent` : "Combined model risk not available yet"}>
        <defs>
          <linearGradient id="gaugeGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="var(--ok)" />
            <stop offset="55%" stopColor="var(--warn)" />
            <stop offset="100%" stopColor="var(--bad)" />
          </linearGradient>
        </defs>
        <path d="M 30 150 A 120 120 0 0 1 270 150" className="gauge-track" pathLength="100" />
        <path d="M 30 150 A 120 120 0 0 1 270 150" className="gauge-fill" pathLength="100" strokeDasharray={`${animated * 100} 100`} />
        <line x1={tx(104)} y1={ty(104)} x2={tx(138)} y2={ty(138)} className="gauge-threshold" />
        <text x={tx(152)} y={ty(152) + 4} className="gauge-threshold-label" textAnchor="middle">{pct(threshold)}</text>
      </svg>
      <div className="gauge-readout">
        <span className="big-num">{available ? Math.round(animated * 100) : "—"}<small>{available ? "%" : ""}</small></span>
        <span className="gauge-caption">MODEL RISK</span>
      </div>
    </div>
  );
}

function StreamRadar({ spoof, scam, voice }) {
  const cx = 160, cy = 112, r = 74;
  const axes = [-90, 30, 150].map((d) => (d * Math.PI) / 180);
  const point = (i, v, scale = 1) => `${cx + r * scale * v * Math.cos(axes[i])},${cy + r * scale * v * Math.sin(axes[i])}`;
  const ring = (s) => [0, 1, 2].map((i) => point(i, 1, s)).join(" ");
  const values = [clamp01(spoof), clamp01(scam), clamp01(voice)];
  const shape = values.map((v, i) => point(i, v)).join(" ");
  const names = ["Spoof", "Scam language", "Voice mismatch"];
  const lab = [
    { x: cx, y: cy - r - 16, a: "middle" },
    { x: cx + r * Math.cos(axes[1]) + 8, y: cy + r * Math.sin(axes[1]) + 22, a: "start" },
    { x: cx + r * Math.cos(axes[2]) - 8, y: cy + r * Math.sin(axes[2]) + 22, a: "end" },
  ];
  return (
    <svg viewBox="0 0 320 230" className="radar" role="img" aria-label="Balance of the three detection streams">
      {[0.33, 0.66, 1].map((s) => (<polygon key={s} points={ring(s)} className="radar-ring" />))}
      {[0, 1, 2].map((i) => (<line key={i} x1={cx} y1={cy} x2={cx + r * Math.cos(axes[i])} y2={cy + r * Math.sin(axes[i])} className="radar-axis" />))}
      <polygon points={shape} className="radar-shape" />
      {values.map((v, i) => (<circle key={i} cx={cx + r * v * Math.cos(axes[i])} cy={cy + r * v * Math.sin(axes[i])} r="4" className="radar-dot" />))}
      {names.map((n, i) => (<text key={n} x={lab[i].x} y={lab[i].y} textAnchor={lab[i].a} className="radar-label">{n}</text>))}
    </svg>
  );
}

function RiskTimeline({ points, threshold }) {
  const W = 640, H = 250, L = 46, R = 18, T = 18, B = 30;
  const [hover, setHover] = useState(null);
  const x = (i) => L + (i / (MAX_POINTS - 1)) * (W - L - R);
  const y = (v) => T + (1 - clamp01(v)) * (H - T - B);
  const line = points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.risk)}`).join(" ");
  const area = points.length ? `${line} L${x(points.length - 1)},${y(0)} L${x(0)},${y(0)} Z` : "";
  const active = hover != null && points[hover] ? hover : null;

  function onMove(e) {
    if (!points.length) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best = 0, bestD = Infinity;
    points.forEach((_, i) => {
      const d = Math.abs(x(i) - px);
      if (d < bestD) { bestD = d; best = i; }
    });
    setHover(best);
  }

  const p = active != null ? points[active] : null;
  const tipLeft = p ? Math.min(78, Math.max(22, (x(active) / W) * 100)) : 0;

  return (
    <div className="timeline">
      {points.length === 0 ? (
        <div className="empty-state">
          <strong>No risk history yet</strong>
          <span>Start monitoring to see the combined risk evolve window by window.</span>
        </div>
      ) : (
        <div className="timeline-wrap">
          <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img" aria-label="Combined risk over the most recent windows">
            <defs>
              <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.28" />
                <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
              </linearGradient>
            </defs>
            {[0, 0.25, 0.5, 0.75, 1].map((g) => (
              <g key={g}>
                <line x1={L} x2={W - R} y1={y(g)} y2={y(g)} className="tl-grid" />
                <text x={L - 10} y={y(g) + 4} textAnchor="end" className="tl-axis">{Math.round(g * 100)}%</text>
              </g>
            ))}
            <line x1={L} x2={W - R} y1={y(threshold)} y2={y(threshold)} className="tl-threshold" />
            <text x={W - R} y={y(threshold) - 6} textAnchor="end" className="tl-threshold-label">Alert threshold {pct(threshold)}%</text>
            <path d={area} fill="url(#areaGrad)" />
            <path d={line} className="tl-line" />
            {active != null && <line x1={x(active)} x2={x(active)} y1={T} y2={H - B} className="tl-cursor" />}
            {points.map((pt, i) => (
              <circle key={i} cx={x(i)} cy={y(pt.risk)} r={i === active ? 6 : i === points.length - 1 ? 5 : 3} className={`tl-point ${pt.alert ? "is-alert" : ""} ${i === points.length - 1 ? "is-last" : ""}`} />
            ))}
            <text x={L} y={H - 8} className="tl-axis">older</text>
            <text x={W - R} y={H - 8} textAnchor="end" className="tl-axis">latest</text>
          </svg>
          {p && (
            <div className="tl-tip" style={{ left: `${tipLeft}%` }}>
              <strong>Window {active + 1} of {points.length}</strong>
              <span>Risk <b>{pct(p.risk)}%</b>{p.alert ? " · alert" : ""}</span>
              <span>Spoof <b>{p.spoof == null ? "—" : `${pct(p.spoof)}%`}</b></span>
              <span>Scam language <b>{p.scam == null ? "—" : `${pct(p.scam)}%`}</b></span>
              <span>Voice mismatch <b>{p.voice == null ? "—" : `${pct(p.voice)}%`}</b></span>
              <span>Latency <b>{p.latency == null ? "—" : `${Math.round(p.latency)} ms`}</b></span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function LatencyWaterfall({ lat }) {
  const stages = [
    { key: "spoof", label: "Audio spoof detection" },
    { key: "asr", label: "Speech recognition (ASR)" },
    { key: "intent", label: "Scam-intent detection" },
    { key: "fusion", label: "Temporal fusion" },
  ];
  const sum = stages.reduce((a, s) => a + (typeof lat?.[s.key] === "number" ? lat[s.key] : 0), 0);
  return (
    <div className="latency">
      <div className="latency-stack" aria-hidden="true">
        {stages.map((s) => {
          const v = lat?.[s.key];
          const w = typeof v === "number" && sum > 0 ? (v / sum) * 100 : 0;
          return <span key={s.key} className={`seg seg-${s.key}`} style={{ width: `${w}%` }} />;
        })}
        {sum === 0 && <span className="seg seg-empty" style={{ width: "100%" }} />}
      </div>
      <ul className="latency-list">
        {stages.map((s) => {
          const v = lat?.[s.key];
          return (
            <li key={s.key}>
              <span className={`swatch seg-${s.key}`} />
              <span className="ll-name">{s.label}</span>
              <strong>{typeof v === "number" ? `${v.toFixed(0)} ms` : "—"}</strong>
            </li>
          );
        })}
      </ul>
      <p className="note">Stage timings may overlap; they are not necessarily additive. Measurements are from the latest processed window.</p>
    </div>
  );
}

function LiveOrb({ analyserRef, recording, silent, tone }) {
  const wrapRef = useRef(null);
  const barsRef = useRef([]);
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!recording) {
      wrap?.style.setProperty("--lvl", "0");
      barsRef.current.forEach((el) => { if (el) el.style.transform = ""; });
      return undefined;
    }
    const buf = new Uint8Array(64);
    let raf;
    const loop = () => {
      const an = analyserRef.current;
      let sum = 0;
      if (an) {
        an.getByteFrequencyData(buf);
        for (let i = 1; i <= 32; i += 1) sum += buf[i];
      }
      const lvl = Math.min(1, (sum / 32 / 255) * 2.4);
      wrap?.style.setProperty("--lvl", lvl.toFixed(3));
      barsRef.current.forEach((el, i) => {
        if (el) el.style.transform = `scaleY(${(0.1 + (buf[i + 1] / 255) * 1.1).toFixed(3)})`;
      });
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => {
      cancelAnimationFrame(raf);
      wrap?.style.setProperty("--lvl", "0");
    };
  }, [recording, analyserRef]);

  return (
    <div ref={wrapRef} className={`orb-wrap tone-${tone} ${!recording ? "is-idle" : silent ? "is-silent" : "is-live"}`}>
      <div className="orb"><span className="orb-core" /><span className="orb-ring" /></div>
      <div className="orb-bars" aria-hidden="true">
        {Array.from({ length: 32 }, (_, i) => (<span key={i} ref={(el) => { barsRef.current[i] = el; }} />))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  App                                                                */
/* ------------------------------------------------------------------ */
export default function App() {
  const [status, setStatus] = useState("Disconnected");
  const [score, setScore] = useState(null);
  const [latency, setLatency] = useState(null);
  const [error, setError] = useState("");
  const [recording, setRecording] = useState(false);
  const [fileResult, setFileResult] = useState(null);
  const [fileLoading, setFileLoading] = useState(false);
  const [fileError, setFileError] = useState("");
  const [riskHistory, setRiskHistory] = useState([]);

  // Live analysis (spoof + speech + scam signals + voice check)
  const [msg, setMsg] = useState(null);
  const [silent, setSilent] = useState(false);
  const [contacts, setContacts] = useState([]);
  const [contact, setContact] = useState("");
  const [label, setLabel] = useState("unknown");

  // Challenge-Response verification
  const [challenge, setChallenge] = useState(null);
  const [challengeLoading, setChallengeLoading] = useState(false);
  const [challengeError, setChallengeError] = useState("");
  const [challengeAnswer, setChallengeAnswer] = useState("");
  const [challengeResult, setChallengeResult] = useState(null);
  const [challengeSubmitting, setChallengeSubmitting] = useState(false);

  // UI-only state (new)
  const [timeline, setTimeline] = useState([]);
  const [summary, setSummary] = useState(null);
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem("rv_theme") || "light"; } catch { return "light"; }
  });
  const [consented, setConsented] = useState(() => {
    try { return localStorage.getItem("rv_consent_v1") === "yes"; } catch { return false; }
  });

  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const contactRef = useRef("");
  const labelRef = useRef("unknown");
  const analyserRef = useRef(null); // UI-only: drives the live audio visual
  const statsRef = useRef(newStats()); // UI-only: session summary counters

  useEffect(() => {
    loadContacts();
    return () => {
      stopAudio();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setChallenge(null);
    setChallengeAnswer("");
    setChallengeResult(null);
    setChallengeError("");

    if (contact && msg?.streams?.fusion?.alert) {
      loadChallenge(contact);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [contact, msg?.streams?.fusion?.alert]);

  async function loadContacts() {
    try {
      const response = await fetch(`${API_URL}/contacts`);
      const data = await response.json();
      setContacts(data.contacts || []);
    } catch {
      // Backend not reachable yet, or it has no /contacts endpoint. The voice check simply stays unavailable.
    }
  }

  function sendContact(name) {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "select_contact", name: name || null }));
    }
  }

  // Kept for the backend's set_label message; the redesigned UI does not expose it.
  // eslint-disable-next-line no-unused-vars
  function onLabelChange(event) {
    const value = event.target.value;
    setLabel(value);
    labelRef.current = value;
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "set_label", label: value }));
    }
  }

  function onContactChange(event) {
    const name = event.target.value;
    setContact(name);
    contactRef.current = name;
    sendContact(name);
  }

  async function loadChallenge(contactName) {
    setChallenge(null);
    setChallengeAnswer("");
    setChallengeResult(null);
    setChallengeError("");

    if (!contactName || !msg?.streams?.fusion?.alert) {
      return;
    }

    setChallengeLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/verification/challenge/${encodeURIComponent(contactName)}`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not retrieve verification question.");
      }

      setChallenge(data);
    } catch (err) {
      setChallengeError(
        err.message || "Unable to load the verification question."
      );
    } finally {
      setChallengeLoading(false);
    }
  }

  async function submitChallengeAnswer() {
    if (!contact || !challengeAnswer.trim()) {
      setChallengeError("Please enter the caller's response.");
      return;
    }

    setChallengeSubmitting(true);
    setChallengeError("");
    setChallengeResult(null);

    try {
      const response = await fetch(
        `${API_URL}/verification/evaluate`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            contact,
            answer: challengeAnswer,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Verification failed.");
      }

      setChallengeResult(data);
      setChallengeAnswer("");
    } catch (err) {
      setChallengeError(
        err.message || "Could not evaluate the caller's response."
      );
    } finally {
      setChallengeSubmitting(false);
    }
  }

  function markChallengeInconclusive() {
    setChallengeAnswer("");
    setChallengeError("");

    setChallengeResult({
      status: "inconclusive",
      verified: false,
      message: "Caller identity could not be verified.",
    });
  }

  async function startAudio() {
    try {
      // Reset previous monitoring session data
      setError("");
      setScore(null);
      setLatency(null);
      setMsg(null);
      setSilent(false);
      setRiskHistory([]);
      setTimeline([]);
      setSummary(null);
      statsRef.current = newStats();
      statsRef.current.startedAt = Date.now();
      setStatus("Requesting microphone…");
      loadContacts();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      streamRef.current = stream;

      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      console.log("Browser audio sample rate:", audioContext.sampleRate);

      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = async () => {
        try {
          setStatus("Connected — listening");
          setRecording(true);
          sendContact(contactRef.current);
          ws.send(JSON.stringify({ type: "set_label", label: labelRef.current }));

          const source = audioContext.createMediaStreamSource(stream);
          sourceRef.current = source;

          // UI-only: a read-only analyser so the interface can show live audio level.
          try {
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 128;
            source.connect(analyser);
            analyserRef.current = analyser;
          } catch {
            analyserRef.current = null;
          }

          await audioContext.audioWorklet.addModule("/processor.js");

          const processor = new AudioWorkletNode(
            audioContext,
            "pcm-processor"
          );
          processorRef.current = processor;

          processor.port.onmessage = (event) => {
            if (ws.readyState !== WebSocket.OPEN) return;

            ws.send(event.data);
          };

          source.connect(processor);

          // Keep the audio graph active without playing microphone audio
          // through the speakers.
          const mute = audioContext.createGain();
          mute.gain.value = 0;
          processor.connect(mute);
          mute.connect(audioContext.destination);
        } catch (err) {
          setError(err.message || "Could not initialize audio processing.");
          setStatus("Unable to start");
          await stopAudio();
        }
      };

      ws.onerror = () => {
        setError("WebSocket connection failed. Check that FastAPI is running.");
        setStatus("Connection error");
      };

      ws.onclose = () => {
        setStatus("Disconnected");
        setRecording(false);
      };

      ws.onmessage = (event) => {
        try {
          const result = JSON.parse(event.data);

          if (result.type === "error") {
            setError(`Server: ${result.detail}`);
            return;
          }
          if (result.type === "contact_set" || result.type === "label_set") return;
          if (result.silent) {
            setSilent(true);
            return;
          }

          if (result.streams) {
            setSilent(false);
            setError("");
            setMsg(result);
            setScore(result.streams.spoof?.score ?? null);
            setLatency(result.latency_ms?.total ?? null);

            const fusionRisk = result.streams.fusion?.risk;

            if (
              typeof fusionRisk === "number" &&
              Number.isFinite(fusionRisk)
            ) {
              setRiskHistory((previous) => [
                ...previous,
                {
                  risk: fusionRisk,
                  timestamp: Date.now(),
                },
              ].slice(-30));
            }

            // UI-only: per-window details for the interactive timeline and the session summary.
            const f = result.streams.fusion;
            const st = statsRef.current;
            st.windows += 1;
            if (typeof f?.risk === "number" && Number.isFinite(f.risk)) {
              st.peak = Math.max(st.peak, f.risk);
              setTimeline((prev) => [
                ...prev,
                {
                  risk: f.risk,
                  alert: !!f.alert,
                  spoof: result.streams.spoof?.score ?? null,
                  scam: maxIntent(result.streams.intent),
                  voice: result.streams.voice ? result.streams.voice.score : null,
                  latency: result.latency_ms?.total ?? null,
                },
              ].slice(-MAX_POINTS));
            }
            if (f?.alert) st.alertWindows += 1;
            if (typeof result.latency_ms?.total === "number") {
              st.latSum += result.latency_ms.total;
              st.latN += 1;
            }
            (f?.evidence || []).forEach((e) => {
              st.signals[e] = (st.signals[e] || 0) + 1;
            });
          } else if (result.stream === "spoof") {
            setScore(result.score);
            setLatency(result.latency_ms);
          }
        } catch {
          setError("Received an invalid response from the backend.");
        }
      };
    } catch (err) {
      setError(err.message || "Could not start microphone.");
      setStatus("Unable to start");
      await stopAudio();
    }
  }

  async function stopAudio() {
    setRecording(false);

    if (processorRef.current) {
      processorRef.current.port.onmessage = null;
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }

    analyserRef.current = null;

    if (wsRef.current) {
      wsRef.current.onclose = null;

      if (
        wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING
      ) {
        wsRef.current.close();
      }

      wsRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      await audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }

    setStatus("Disconnected");
  }

  // UI-only wrapper: snapshot the session summary, then stop exactly as before.
  async function handleStop() {
    const st = statsRef.current;
    if (st.windows > 0) {
      setSummary({
        windows: st.windows,
        peakRisk: st.peak,
        alertWindows: st.alertWindows,
        avgLatencyMs: st.latN ? st.latSum / st.latN : null,
        durationSec: st.startedAt ? Math.round((Date.now() - st.startedAt) / 1000) : null,
        signals: st.signals,
      });
    }
    await stopAudio();
  }

  function downloadSummary() {
    if (!summary) return;
    const data = {
      generated_at: new Date().toISOString(),
      note: "Scores only. No audio and no transcript are included.",
      windows_processed: summary.windows,
      peak_model_risk_percent: pct(summary.peakRisk),
      alert_windows: summary.alertWindows,
      average_latency_ms: summary.avgLatencyMs == null ? null : Math.round(summary.avgLatencyMs),
      duration_seconds: summary.durationSec,
      evidence_counts: summary.signals,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "rakshavoice-session-summary.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  async function analyzeFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileResult(null);
    setFileError("");

    if (!/\.(wav|ogg|oga|opus)$/i.test(file.name)) {
      setFileError("Please select a WAV, OGG, OGA, or OPUS audio file.");
      event.target.value = "";
      return;
    }

    setFileLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(
        `${API_URL}/analyze-file`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Audio analysis failed.");
      }

      setFileResult(data);
    } catch (err) {
      setFileError(
        err.message || "Could not connect to the backend."
      );
    } finally {
      setFileLoading(false);
      event.target.value = "";
    }
  }

  /* ---------------------------- derived values --------------------------- */
  const percent = score == null ? null : Math.round(score * 10000) / 100;

  const intent = msg?.streams?.intent;
  const voice = msg?.streams?.voice;
  const fusion = msg?.streams?.fusion;
  const explanation = msg?.streams?.explanation;
  const fired = new Set(intent?.evidence || []);
  const lat = msg?.latency_ms;
  const voiceMismatch = !!voice?.evidence?.length;
  const voicePercent = voice ? Math.round(voice.score * 100) : 0;
  const topIntent = maxIntent(intent);

  const fusionAvailable = !!fusion?.available;
  const risk = fusion?.risk ?? 0;
  const threshold = fusion?.threshold ?? 0.7;
  const alertOn = !!fusion?.alert;
  const consecutive = fusion?.consecutive_high_windows ?? 0;
  const tone = fusionAvailable ? riskTone(risk, threshold, alertOn) : "low";
  const latestRisk = riskHistory.length ? riskHistory[riskHistory.length - 1].risk : null;

  const statusClass = recording
    ? "active"
    : status.toLowerCase().includes("error") || status.toLowerCase().includes("unable")
      ? "error"
      : status.toLowerCase().includes("requesting") || status.toLowerCase().includes("connecting")
        ? "connecting"
        : "";
  const transcriptText =
    typeof msg?.transcript === "string" &&
    msg.transcript.trim() !== "" &&
    !["nan", "null", "undefined"].includes(msg.transcript.trim().toLowerCase())
      ? msg.transcript.trim()
      : "";

  /* ------------------------------ UI effects ----------------------------- */
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("rv_theme", theme); } catch { /* ignore */ }
  }, [theme]);

  useEffect(() => {
    document.documentElement.dataset.alert = alertOn ? "true" : "false";
  }, [alertOn]);

  useEffect(() => {
    function onKey(e) {
      if (e.key.toLowerCase() !== "m" || e.metaKey || e.ctrlKey || e.altKey) return;
      const tag = (e.target?.tagName || "").toLowerCase();
      if (["input", "textarea", "select"].includes(tag) || !consented) return;
      if (recording) handleStop(); else startAudio();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recording, consented]);

  function acceptConsent() {
    setConsented(true);
    try { localStorage.setItem("rv_consent_v1", "yes"); } catch { /* ignore */ }
  }

  /* -------------------------------- render ------------------------------- */
  return (
    <div className={`page tone-${tone} ${alertOn ? "page-alert" : ""}`}>
      {!consented && (
        <div className="overlay" role="dialog" aria-modal="true" aria-labelledby="consent-title">
          <div className="modal">
            <div className="modal-icon"><Icon name="lock" /></div>
            <h2 id="consent-title">Before you start</h2>
            <p>
              RakshaVoice analyses the call through your microphone while it happens. Audio is
              processed in memory on this computer and discarded after each window. No recording
              is created, and the live transcript is not stored.
            </p>
            <ul>
              <li>Scores are model predictions, not proof of fraud.</li>
              <li>Enrol a person&apos;s voice only with their consent.</li>
              <li>Use it as a warning to help your own decision.</li>
            </ul>
            <button className="btn btn-dark" onClick={acceptConsent}>I understand</button>
          </div>
        </div>
      )}

      {summary && (
        <div className="overlay" role="dialog" aria-modal="true" aria-labelledby="summary-title">
          <div className="modal modal-wide">
            <p className="eyebrow">SESSION SUMMARY</p>
            <h2 id="summary-title">Monitoring stopped</h2>
            <div className="summary-grid">
              <div><span>Windows processed</span><strong>{summary.windows}</strong></div>
              <div><span>Peak model risk</span><strong>{pct(summary.peakRisk)}%</strong></div>
              <div><span>Alert windows</span><strong>{summary.alertWindows}</strong></div>
              <div><span>Average latency</span><strong>{summary.avgLatencyMs == null ? "—" : `${Math.round(summary.avgLatencyMs)} ms`}</strong></div>
              <div><span>Duration</span><strong>{summary.durationSec == null ? "—" : `${summary.durationSec} s`}</strong></div>
              <div>
                <span>Signals seen</span>
                <strong className="summary-signals">
                  {Object.keys(summary.signals).length
                    ? Object.entries(summary.signals).map(([k, n]) => `${EVIDENCE_NAMES[k] || k} ×${n}`).join(", ")
                    : "None"}
                </strong>
              </div>
            </div>
            <p className="note">Contains scores only. No audio and no transcript are saved or included in the download.</p>
            <div className="modal-actions">
              <button className="btn btn-ghost" onClick={downloadSummary}>Download JSON</button>
              <button className="btn btn-dark" onClick={() => setSummary(null)}>Close</button>
            </div>
          </div>
        </div>
      )}

      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <span className="brand-mark"><Icon name="shield" /></span>
            <span className="brand-name">RakshaVoice</span>
          </div>

          <div className={`status-pill ${statusClass}`} aria-live="polite">
            <span className="dot" />
            {recording ? "Connected" : status}
          </div>

          <div className="topbar-actions">
            <button className="icon-btn" onClick={() => setTheme(theme === "light" ? "dark" : "light")} aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`} title="Toggle theme">
              <Icon name={theme === "light" ? "moon" : "sun"} />
            </button>
            {!recording ? (
              <button className="btn btn-dark" onClick={startAudio} title="Shortcut: M">Start monitoring</button>
            ) : (
              <button className="btn btn-danger" onClick={handleStop} title="Shortcut: M">Stop monitoring</button>
            )}
          </div>
        </div>
      </header>

      <main className="shell">
        <section className="intro">
          <h1>Live call monitor</h1>
          <p>
            Listens through your microphone, checks whether the voice is synthetic, reads what is
            being asked, and compares the voice with a person you enrolled.
          </p>
          <p className="privacy-line"><Icon name="lock" /> Audio is processed in memory and discarded after analysis.</p>
          {error && <p className="error" role="alert">{error}</p>}
        </section>

        {alertOn && (
          <section className="alert-banner" role="alert" aria-live="assertive">
            <span className="alert-mark">!</span>
            <div className="alert-body">
              <p className="eyebrow">SECURITY ALERT</p>
              <h2>Elevated call risk detected</h2>
              {explanation?.available ? (
                <>
                  <p>{explanation.summary}</p>
                  <div className="alert-action">
                    <strong>Recommended action</strong>
                    <p>{explanation.recommended_action}</p>
                  </div>
                </>
              ) : (
                <p>
                  Multiple elevated-risk windows have been detected. Review the transcript and
                  available evidence before taking action.
                </p>
              )}
            </div>
          </section>
        )}

        <div className="grid">
          {/* HERO RISK */}
          <section className="card hero span-8">
            <div className="card-head">
              <div>
                <p className="eyebrow">TEMPORAL FUSION MODEL</p>
                <h2>Combined call risk</h2>
              </div>
              <span className={`badge ${alertOn ? "badge-bad" : ""}`}>{alertOn ? "Elevated risk alert" : "Monitoring"}</span>
            </div>

            <div className="hero-body">
              <RiskGauge value={risk} threshold={threshold} tone={tone} available={fusionAvailable} />

              <div className="hero-side">
                <p className={`state-line state-${tone}`}>
                  {!fusionAvailable
                    ? "Waiting for data"
                    : alertOn
                      ? "Elevated Risk Alert"
                      : risk >= threshold
                        ? "Threshold Reached"
                        : "Below Alert Threshold"}
                </p>

                <div className="persistence" aria-label="Risk persistence">
                  <div className="persistence-top">
                    <strong>Risk persistence</strong>
                    <span>{consecutive} consecutive high</span>
                  </div>
                  <div className="steps">
                    {[1, 2, 3].map((stage) => {
                      const active = consecutive >= stage;
                      return (
                        <div key={stage} className={`step ${active ? (stage >= 2 ? "step-alert" : "step-on") : ""}`}>
                          <span className="step-dot" />
                          <span>{stage === 3 ? "3+" : stage}</span>
                        </div>
                      );
                    })}
                  </div>
                  <p className="note">
                    {alertOn
                      ? "Alert threshold reached: multiple consecutive high-risk windows."
                      : "An alert needs 2 consecutive high-risk windows."}
                  </p>
                </div>

                <dl className="facts">
                  <div><dt>Latest risk</dt><dd>{latestRisk == null ? "—" : `${pct(latestRisk)}%`}</dd></div>
                  <div><dt>Windows used</dt><dd>{fusionAvailable ? `${fusion.windows_seen}/24` : "—"}</dd></div>
                  <div><dt>Threshold</dt><dd>{pct(threshold)}%</dd></div>
                </dl>
              </div>
            </div>

            {!fusionAvailable && (
              <p className="note">
                {fusion?.error
                  ? `Fusion model unavailable: ${fusion.error}. Check fusion/models/fusion_gru.pt and fusion_config.json.`
                  : "Start monitoring to receive a combined risk score."}
              </p>
            )}
            <p className="note fine">
              Research prototype only. This score is not a verified probability of fraud and should
              not be used as the sole basis for decisions.
            </p>
          </section>

          {/* LIVE AUDIO */}
          <section className="card live span-4">
            <div className="card-head">
              <div>
                <p className="eyebrow">LIVE AUDIO</p>
                <h2>Microphone</h2>
              </div>
            </div>
            <LiveOrb analyserRef={analyserRef} recording={recording} silent={silent} tone={tone} />
            <p className="live-status">
              {!recording ? "Microphone idle" : silent ? "Listening · silence detected" : "Audio activity detected"}
            </p>
            <p className="note center">
              {!recording
                ? "Press Start monitoring (or the M key) to begin."
                : silent
                  ? "Waiting for incoming speech or audio."
                  : "The microphone is receiving audio."}
            </p>
          </section>

          {/* EVIDENCE EXPLAINER BAR */}
          <section className="explain span-12" aria-live="polite">
            <div className="explain-head">
              <Icon name="spark" />
              <span>Evidence explainer</span>
            </div>
            <div className="explain-box">
              {alertOn && explanation?.available ? (
                <p><b>{explanation.summary}</b> <span className="explain-action">{explanation.recommended_action}</span></p>
              ) : alertOn ? (
                <p>Multiple elevated-risk windows detected. Review the transcript and evidence below.</p>
              ) : (
                <p className="muted">A plain-language reason and a recommended action appear here when a sustained high-risk pattern is detected.</p>
              )}
              <div className="chips">
                {(fusion?.evidence || []).length ? (
                  fusion.evidence.map((item) => (<span key={item} className="chip on">{EVIDENCE_NAMES[item] || item}</span>))
                ) : (
                  <span className="chip">{fusionAvailable ? "No thresholded evidence in latest window" : "No evidence yet"}</span>
                )}
              </div>
            </div>
          </section>

          {/* STREAM CARDS */}
          <article className="card stream span-4">
            <div className="stream-top"><span className="stream-icon"><Icon name="wave" /></span><p className="eyebrow">AUDIO AUTHENTICITY</p></div>
            <div className="stream-value">{percent == null ? "—" : `${percent.toFixed(2)}%`}</div>
            <p className="muted">Predicted spoof probability</p>
            <div className="bar"><span style={{ width: `${percent == null ? 0 : Math.min(100, percent)}%` }} /></div>
            <p className="foot">{score == null ? "Waiting for audio" : score >= 0.5 ? "Higher spoof signal" : "Lower spoof signal"}</p>
            <p className="fine-note">Experimental: not yet calibrated for live microphone audio.</p>
          </article>

          <article className="card stream span-4">
            <div className="stream-top"><span className="stream-icon"><Icon name="text" /></span><p className="eyebrow">SCAM LANGUAGE</p></div>
            <div className="stream-value">{topIntent == null ? "—" : `${Math.round(topIntent * 100)}%`}</div>
            <p className="muted">Highest intent signal in latest window</p>
            <div className="bar"><span style={{ width: `${topIntent == null ? 0 : topIntent * 100}%` }} /></div>
            <p className="foot">{fired.size ? `${fired.size} evidence indicator(s) fired` : "No thresholded evidence"}</p>
          </article>

          <article className="card stream span-4">
            <div className="stream-top"><span className="stream-icon"><Icon name="user" /></span><p className="eyebrow">VOICE MATCH</p></div>
            <div className="stream-value stream-text">{voice ? (voiceMismatch ? "Mismatch" : "Match") : "—"}</div>
            <p className="muted">Optional enrolled-speaker comparison</p>
            <div className="bar"><span className={voiceMismatch ? "bar-bad" : ""} style={{ width: `${voice ? voicePercent : 0}%` }} /></div>
            <p className="foot">{voice ? `Similarity ${voice.similarity}` : "Select an enrolled contact"}</p>
          </article>

          {/* TIMELINE + RADAR */}
          <section className="card span-8">
            <div className="card-head">
              <div>
                <p className="eyebrow">TEMPORAL RISK ANALYSIS</p>
                <h2>Risk over time</h2>
                <p className="muted">Hover the chart to inspect each window. Recent combined-risk measurements from the fusion model.</p>
              </div>
              <div className="head-stat"><span>Latest risk</span><strong>{latestRisk == null ? "—" : `${pct(latestRisk)}%`}</strong></div>
            </div>
            <RiskTimeline points={timeline} threshold={threshold} />
            <p className="note">{timeline.length} / {MAX_POINTS} windows</p>
          </section>

          <section className="card span-4">
            <div className="card-head">
              <div>
                <p className="eyebrow">STREAM BALANCE</p>
                <h2>Three signals</h2>
                <p className="muted">How strongly each stream fired in the latest window.</p>
              </div>
            </div>
            <StreamRadar spoof={score} scam={topIntent} voice={voice ? voice.score : 0} />
          </section>

          {/* TRANSCRIPT + SIGNALS */}
          <section className="card span-8">
            <div className="card-head">
              <div>
                <p className="eyebrow">LIVE TRANSCRIPT</p>
                <h2>What the caller is saying</h2>
              </div>
              <span className={`badge ${!recording ? "" : transcriptText ? "badge-ok" : ""}`}>
                {!recording ? "Stopped" : transcriptText ? "Speech detected" : silent ? "Silence" : "Listening"}
              </span>
            </div>
            <div className={`transcript ${transcriptText ? "has-text" : ""}`}>
              {transcriptText
                ? transcriptText
                : recording
                  ? silent
                    ? "Listening… (silence)"
                    : "Waiting for speech…"
                  : "Start monitoring to see the live transcript."}
            </div>
            <p className="note">Text is shown here only and is not stored.</p>

            <p className="eyebrow signals-label">SCAM SIGNALS</p>
            <div className="chips">
              {LABELS.map((lb) => {
                const value = intent?.labels?.[lb];
                const on = fired.has(lb);
                return (
                  <span key={lb} className={`chip ${on ? "on" : ""}`}>
                    {LABEL_NAMES[lb]}
                    {value !== undefined ? ` · ${Math.round(value * 100)}%` : ""}
                  </span>
                );
              })}
            </div>
          </section>

          {/* VOICE VERIFICATION */}
          <section className="card span-4">
            <div className="card-head">
              <div>
                <p className="eyebrow">VOICE IDENTITY</p>
                <h2>Voice verification</h2>
              </div>
              <span className="badge">Optional</span>
            </div>
            <label className="field-label" htmlFor="contact">The caller says they are</label>
            <select id="contact" className="select" value={contact} onChange={onContactChange}>
              <option value="">Nobody in particular</option>
              {contacts.map((name) => (<option key={name} value={name}>{name}</option>))}
            </select>

            {voice ? (
              <>
                <div className={`verdict ${voiceMismatch ? "bad" : "good"}`}>
                  {voiceMismatch ? "Voice does not match" : "Voice matches"}
                </div>
                <div className="bar">
                  <span className={voiceMismatch ? "bar-bad" : "bar-ok"} style={{ width: `${voicePercent}%` }} />
                </div>
                <div className="bar-labels"><span>Same person</span><span>Different person</span></div>
                <p className="muted">Compared with “{voice.enrolled}” · similarity {voice.similarity}</p>
              </>
            ) : (
              <p className="muted">
                {contacts.length
                  ? "Pick the person the caller claims to be, then keep speaking."
                  : "No enrolled contacts yet. Enrol someone with the enrolment script first."}
              </p>
            )}
          </section>

          {/* LATENCY */}
          <section className="card span-12">
            <div className="card-head">
              <div>
                <p className="eyebrow">PROCESSING PERFORMANCE</p>
                <h2>Inference latency</h2>
                <p className="muted">Time taken by each analysis stage in the latest audio window.</p>
              </div>
              <div className="head-stat"><span>Total latency</span><strong>{typeof lat?.total === "number" ? `${lat.total.toFixed(0)} ms` : latency != null ? `${Math.round(latency)} ms` : "—"}</strong></div>
            </div>
            <LatencyWaterfall lat={lat} />
          </section>

          {/* CHALLENGE-RESPONSE */}
          <section className={`card challenge span-6 ${alertOn ? "challenge-live" : ""}`}>
            <div className="card-head">
              <div>
                <p className="eyebrow">ACTIVE VERIFICATION</p>
                <h2>Challenge–response check</h2>
              </div>
              <span className="badge badge-glass">Security check</span>
            </div>

            {!contact ? (
              <p className="lead">Select a contact in the Voice verification panel to enable verification.</p>
            ) : !alertOn ? (
              <p className="lead">Challenge–response becomes available when an elevated-risk alert is detected.</p>
            ) : challengeLoading ? (
              <p className="lead">Retrieving verification question...</p>
            ) : challengeError && !challenge ? (
              <p className="error on-dark">{challengeError}</p>
            ) : challenge && !challenge.available ? (
              <p className="lead">{challenge.message}</p>
            ) : challenge?.available ? (
              <>
                <div className="challenge-question">
                  <p className="eyebrow">ASK THE CALLER</p>
                  <h3>{challenge.question}</h3>
                  <p className="muted">Ask the question verbally. Enter the caller&apos;s response below.</p>
                </div>

                <label className="field-label" htmlFor="challenge-answer">Caller&apos;s response</label>
                <input
                  id="challenge-answer"
                  className="input"
                  type="password"
                  autoComplete="off"
                  value={challengeAnswer}
                  onChange={(event) => setChallengeAnswer(event.target.value)}
                  placeholder="Enter caller's response"
                  disabled={challengeSubmitting || !!challengeResult}
                />

                <div className="btn-row">
                  <button className="btn btn-light" onClick={submitChallengeAnswer} disabled={challengeSubmitting || !!challengeResult || !challengeAnswer.trim()}>
                    {challengeSubmitting ? "Checking..." : "Evaluate response"}
                  </button>
                  <button className="btn btn-outline-light" onClick={markChallengeInconclusive} disabled={challengeSubmitting || !!challengeResult}>
                    Mark inconclusive
                  </button>
                </div>

                {challengeError && <p className="error on-dark">{challengeError}</p>}

                {challengeResult && (
                  <div className={`challenge-result ${challengeResult.status}`}>
                    <strong>
                      {challengeResult.status === "verified"
                        ? "Identity response verified"
                        : challengeResult.status === "failed"
                          ? "Verification failed"
                          : "Verification inconclusive"}
                    </strong>
                    <p>{challengeResult.message}</p>
                  </div>
                )}

                <p className="note on-dark">
                  The answer is checked locally. A matching answer is an additional verification
                  signal, not proof of identity.
                </p>
              </>
            ) : (
              <p className="lead">Waiting for verification information.</p>
            )}
          </section>

          {/* FILE ANALYSIS */}
          <details className="card file span-6" open={Boolean(fileLoading || fileResult || fileError) || undefined}>
            <summary>
              <span className="file-icon"><Icon name="upload" /></span>
              <div>
                <p className="eyebrow">FILE-BASED ANALYSIS</p>
                <h2>Analyze an audio recording</h2>
                <p className="muted">Upload a recording for offline spoof detection.</p>
              </div>
              <span className="badge">Audio upload</span>
            </summary>

            <div className="file-body">
              <label className={`dropzone ${fileLoading ? "disabled" : ""}`}>
                <input type="file" accept=".wav,.ogg,.oga,.opus,audio/*" onChange={analyzeFile} disabled={fileLoading} />
                <Icon name="upload" />
                <strong>{fileLoading ? "Analyzing audio..." : "Choose an audio file"}</strong>
                <span>Select a recording from your computer</span>
                <span className="fine-note">Supported formats: WAV, OGG, OGA, OPUS</span>
              </label>

              {fileLoading && <p className="note">Processing audio with RakshaVoice...</p>}
              {fileError && <p className="error">{fileError}</p>}

              {fileResult && (
                <div className="file-result">
                  <div className="file-result-head">
                    <div>
                      <p className="eyebrow">ANALYSIS COMPLETE</p>
                      <h3>Detection result</h3>
                    </div>
                    <span className="badge badge-ok">Processed</span>
                  </div>
                  <dl className="file-grid">
                    <div><dt>File name</dt><dd>{fileResult.filename}</dd></div>
                    <div><dt>Duration</dt><dd>{fileResult.duration_seconds} seconds</dd></div>
                    <div><dt>Spoof probability</dt><dd>{fileResult.spoof_probability_percent}%</dd></div>
                    <div><dt>Sample rate</dt><dd>{fileResult.sample_rate} Hz</dd></div>
                  </dl>
                  <p className="note">
                    Higher scores indicate higher predicted spoof probability. This is a model
                    prediction, not a definitive verdict.
                  </p>
                </div>
              )}
            </div>
          </details>
        </div>

        <footer className="footer">
          <p>
            Scores are model predictions, not proof that a voice is authentic or spoofed. Validate
            against known audio before relying on decisions.
          </p>
          <p>RakshaVoice · Local prototype</p>
        </footer>
      </main>
    </div>
  );
}
