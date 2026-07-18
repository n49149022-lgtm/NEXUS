// ============================================================================
// NEXUS MARKETING AGENTS v5.0 // DENO DEPLOY EDITION (STRICT TS SAFE)
// ============================================================================

const kv = await Deno.openKv();

// --- ТИПЫ ---
interface License { key: string; plan: string; expiresAt: string; deviceFingerprint: string | null; activatedAt: string; userId: string | null; }
interface AccountSE { id: string; fio: string; inn: string; email: string; phone: string; regDate: string; }
interface AccountIP { id: string; name: string; inn: string; ogrnip: string; email: string; phone: string; account: string; bank: string; bik: string; address: string; }

// --- УТИЛИТЫ ---
const corsHeaders = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS", "Access-Control-Allow-Headers": "Content-Type, Authorization" };
const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json", ...corsHeaders } });
const uuid = () => crypto.randomUUID();

// --- МОДУЛЬ 1: АНОНИМИЗАЦИЯ ---
const Anonymizer = {
  patterns: [
    { regex: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g, repl: "[EMAIL]" },
    { regex: /(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}/g, repl: "[PHONE]" },
    { regex: /\b\d{10,12}\b/g, repl: "[INN]" },
    { regex: /\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b/g, repl: "[CARD]" },
    { regex: /\b\d{2}\.\d{2}\.\d{4}\b/g, repl: "[DATE]" },
  ],
  anonymize(text: string): string {
    return this.patterns.reduce((acc, p) => acc.replace(p.regex, p.repl), text);
  }
};

// --- МОДУЛЬ 2: RAG (TF-IDF) ---
const RAGEngine = {
  tokenize: (text: string) => text.toLowerCase().match(/\b\w+\b/g) || [],
  
  async buildIndex(docId: string, text: string) {
    const tokens = RAGEngine.tokenize(text);
    const tf: Record<string, number> = {};
    tokens.forEach(t => { tf[t] = (tf[t] || 0) + 1; });
    
    const vals = Object.values(tf) as number[];
    const maxTf = vals.length > 0 ? Math.max(...vals) : 1;
    const tfNorm: Record<string, number> = {};
    for (const [token, count] of Object.entries(tf)) tfNorm[token] = count / maxTf;

    const meta = await kv.get<{ N: number }>(["meta", "rag"]);
    await kv.set(["meta", "rag"], { N: (meta.value?.N || 0) + 1 });

    for (const token of new Set(tokens)) {
      const idx = await kv.get<{ docIds: string[] }>(["index", token]);
      const docIds = idx.value?.docIds || [];
      if (!docIds.includes(docId)) {
        docIds.push(docId);
        await kv.set(["index", token], { docIds });
      }
    }
    await kv.set(["documents", docId], { text: text.substring(0, 50000), tf: tfNorm });
  },

  async search(query: string, topK = 3) {
    const qTokens = RAGEngine.tokenize(query);
    const qTf: Record<string, number> = {};
    qTokens.forEach(t => { qTf[t] = (qTf[t] || 0) + 1; });
    
    const meta = await kv.get<{ N: number }>(["meta", "rag"]);
    const N = meta.value?.N || 1;
    const qVals = Object.values(qTf) as number[];
    const maxQTf = qVals.length > 0 ? Math.max(...qVals) : 1;

    const qVector: Record<string, number> = {};
    for (const [token, count] of Object.entries(qTf)) {
      const idx = await kv.get<{ docIds: string[] }>(["index", token]);
      const df = idx.value?.docIds.length || 1;
      qVector[token] = (count / maxQTf) * Math.log(N / df);
    }

    const candidates = new Set<string>();
    for (const token of Object.keys(qVector)) {
      const idx = await kv.get<{ docIds: string[] }>(["index", token]);
      idx.value?.docIds.forEach(id => candidates.add(id));
    }

    const scores: { docId: string; score: number; snippet: string }[] = [];
    for (const docId of candidates) {
      const doc = await kv.get<{ text: string; tf: Record<string, number> }>(["documents", docId]);
      if (!doc.value) continue;

      let dot = 0, normQ = 0, normD = 0;
      for (const [token, qVal] of Object.entries(qVector)) {
        const dVal = doc.value.tf[token] || 0;
        dot += qVal * dVal;
        normQ += qVal ** 2;
        normD += dVal ** 2;
      }

      const sim = (normQ > 0 && normD > 0) ? dot / (Math.sqrt(normQ) * Math.sqrt(normD)) : 0;
      if (sim > 0) scores.push({ docId, score: Number(sim.toFixed(4)), snippet: doc.value.text.substring(0, 150) + "..." });
    }
    return scores.sort((a, b) => b.score - a.score).slice(0, topK);
  }
};

// --- ОБРАБОТЧИК ЗАПРОСОВ ---
async function handleRequest(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });
  const url = new URL(req.url);
  const path = url.pathname;

  try {
    if (req.method === "GET" && path === "/api/health") return json({ status: "ok", runtime: "Deno Deploy" });

    if (req.method === "POST" && path === "/api/licenses/activate") {
      const body = await req.json() as { key: string; device_fingerprint: string };
      const key = body.key.toUpperCase().trim();
      const isDemo = key === "NEXUS-PRO-2024-KEY" || key === "NEXUS-ENT-2024-KEY";
      const isValidFormat = /^[A-Z0-9]{5}-[A-Z0-9]{5}-[A-Z0-9]{5}-[A-Z0-9]{5}$/.test(key);
      
      if (!isDemo && !isValidFormat) return json({ error: "INVALID_KEY_FORMAT" }, 400);

      const plan = isDemo ? (key.includes("PRO") ? "pro" : "enterprise") : "pro";
      const days = isDemo ? 30 : 7;
      const expiresAt = new Date(Date.now() + days * 86400000).toISOString();
      const license: License = { key, plan, expiresAt, deviceFingerprint: body.device_fingerprint, activatedAt: new Date().toISOString(), userId: null };
      
      await kv.set(["licenses", key], license);
      return json({ plan: license.plan, key: license.key, expiresAt: license.expiresAt, activatedAt: license.activatedAt });
    }

    if (req.method === "GET" && path.startsWith("/api/license/status")) {
      const key = url.searchParams.get("key");
      if (!key) return json({ valid: false, error: "No key" }, 400);
      const res = await kv.get<License>(["licenses", key]);
      if (!res.value || new Date(res.value.expiresAt) < new Date()) return json({ valid: false, error: "Expired" });
      return json({ valid: true, plan: res.value.plan, expiresAt: res.value.expiresAt });
    }

    if (req.method === "POST" && path === "/api/accounts/selfemployed") {
      const body = await req.json() as AccountSE;
      const id = uuid();
      await kv.set(["accounts_se", id], { ...body, id });
      return json({ status: "success", data: { ...body, id } });
    }
    
    if (req.method === "GET" && path === "/api/accounts/selfemployed") {
      const iter = kv.list<AccountSE>({ prefix: ["accounts_se"] }, { limit: 1, reverse: true });
      return json({ data: (await iter.next()).value || null });
    }

    if (req.method === "POST" && path === "/api/accounts/ip") {
      const body = await req.json() as AccountIP;
      const id = uuid();
      await kv.set(["accounts_ip", id], { ...body, id });
      return json({ status: "success", data: { ...body, id } });
    }

    if (req.method === "GET" && path === "/api/accounts/ip") {
      const iter = kv.list<AccountIP>({ prefix: ["accounts_ip"] }, { limit: 1, reverse: true });
      return json({ data: (await iter.next()).value || null });
    }

    if (req.method === "DELETE" && path === "/api/accounts/all") {
      for await (const entry of kv.list({ prefix: ["accounts_se"] })) await kv.delete(entry.key);
      for await (const entry of kv.list({ prefix: ["accounts_ip"] })) await kv.delete(entry.key);
      return json({ status: "success", message: "Purged" });
    }

    if (req.method === "POST" && path === "/api/missions") {
      const body = await req.json();
      const id = uuid();
      await kv.set(["missions", id], { ...body, id });
      return json({ status: "success", id });
    }

    if (req.method === "POST" && path === "/api/anonymize") {
      const { text } = await req.json() as { text: string };
      return json({ original_length: text.length, anonymized: Anonymizer.anonymize(text) });
    }

    if (req.method === "POST" && path === "/api/index/build") {
      const { doc_id, text } = await req.json() as { doc_id: string; text: string };
      await RAGEngine.buildIndex(doc_id, text);
      return json({ status: "success", doc_id });
    }

    if (req.method === "POST" && path === "/api/search") {
      const { query, top_k = 3 } = await req.json() as { query: string; top_k?: number };
      return json({ results: await RAGEngine.search(query, top_k) });
    }

    if (req.method === "POST" && path === "/api/logs/parse") {
      const { log_text } = await req.json() as { log_text: string };
      return json({ parsed: "mock_parsed_data", entriesCount: log_text.split("\n").length });
    }

    if (req.method === "POST" && path === "/api/payments/create") {
      const body = await req.json() as { plan: string; amount: number | string };
      const amount = typeof body.amount === "string" ? parseFloat(body.amount) : (body.amount || 4999);
      const paymentId = `pay_${uuid().replace(/-/g, "").substring(0, 16)}`;
      const licenseKey = `NEXUS-${(body.plan || "pro").toUpperCase()}-${uuid().replace(/-/g, "").substring(0, 8).toUpperCase()}`;
      const expiresAt = new Date(Date.now() + 30 * 86400000).toISOString();
      
      await kv.set(["payments", paymentId], { paymentId, amount, plan: body.plan, status: "succeeded" });
      await kv.set(["licenses", licenseKey], { key: licenseKey, plan: body.plan, expiresAt, deviceFingerprint: null, activatedAt: new Date().toISOString(), userId: null });
      
      return json({ confirmed: true, payment_id: paymentId, license: { key: licenseKey, expires_at: expiresAt } });
    }

    if (req.method === "GET" && path.startsWith("/api/export/")) {
      return json({ type: path.split("/").pop(), timestamp: new Date().toISOString(), data: "mock_export" });
    }

    if (req.method === "POST" && path === "/api/documents/upload") {
      const formData = await req.formData();
      const file = formData.get("file");
      const docId = (formData.get("doc_id") as string) || uuid();
      if (!file || !(file instanceof File)) return json({ error: "No file" }, 400);
      
      const text = await file.text();
      if (text.length > 50000) return json({ error: "File too large for Deno KV (max 50KB)" }, 413);
      
      await RAGEngine.buildIndex(docId, text);
      return json({ status: "success", doc_id: docId, size: text.length });
    }

    return json({ error: "Not Found", path }, 404);
  } catch (err: unknown) {
    console.error("API Error:", err);
    return json({ error: "Internal Server Error", detail: String(err) }, 500);
  }
}

// Fixed strict TS types for Deno Deploy build - v2
Deno.serve(handleRequest);
