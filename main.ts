// ============================================================================
// NEXUS MARKETING AGENTS v5.0 // DENO DEPLOY EDITION (STRICT TS FIXED)
// Pure TypeScript, Deno KV, No LLM, Mathematical RAG (TF-IDF)
// ============================================================================

const kv = await Deno.openKv();

// --- ТИПЫ ДАННЫХ ---
interface License {
  key: string;
  plan: "free" | "pro" | "enterprise";
  expiresAt: string;
  deviceFingerprint: string | null;
  activatedAt: string;
  userId: string | null;
}

interface AccountSE {
  id: string;
  fio: string;
  inn: string;
  email: string;
  phone: string;
  regDate: string;
}

interface AccountIP {
  id: string;
  name: string;
  inn: string;
  ogrnip: string;
  email: string;
  phone: string;
  account: string;
  bank: string;
  bik: string;
  address: string;
}

interface Mission {
  id: string;
  type: string;
  agentId: string;
  results: unknown;
  timestamp: string;
}

// --- УТИЛИТЫ ---
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders },
  });
}

function generateId(): string {
  return crypto.randomUUID();
}

// --- МОДУЛЬ 1: АНОНИМИЗАЦИЯ (Regex) ---
const Anonymizer = {
  patterns: [
    { regex: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g, repl: "[EMAIL]" },
    { regex: /(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}/g, repl: "[PHONE]" },
    { regex: /\b(?:\d{10}|\d{12})\b/g, repl: "[INN]" },
    { regex: /\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b/g, repl: "[CARD]" },
    { regex: /\b\d{4}\s?\d{6}\b/g, repl: "[PASSPORT]" },
    { regex: /\b\d{2}\.\d{2}\.\d{4}\b/g, repl: "[DATE]" },
    { regex: /\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.[А-ЯЁ]\.|\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\b/g, repl: "[FIO]" },
  ],
  anonymize(text: string): string {
    return this.patterns.reduce((acc, { regex, repl }) => acc.replace(regex, repl), text);
  },
};

// --- МОДУЛЬ 2: RAG-ПОИСК (TF-IDF + Cosine Similarity, БЕЗ LLM) ---
const RAGEngine = {
  async tokenize(text: string): Promise<string[]> {
    return text.toLowerCase().match(/\b\w+\b/g) || [];
  },

  async buildIndex(docId: string, text: string): Promise<void> {
    const tokens = await this.tokenize(text);
    const tf: Record<string, number> = {};
    tokens.forEach((t) => { tf[t] = (tf[t] || 0) + 1; });
    
    // ИСПРАВЛЕНИЕ 1: Явное приведение типов для Math.max
    const tfValues = Object.values(tf);
    const maxTf = tfValues.length > 0 ? Math.max(...tfValues) : 1;
    
    const tfNorm: Record<string, number> = {};
    for (const [token, count] of Object.entries(tf)) {
      tfNorm[token] = count / maxTf;
    }

    const metaRes = await kv.get<{ N: number }>(["meta", "rag"]);
    const N = (metaRes.value?.N || 0) + 1;
    await kv.set(["meta", "rag"], { N });

    for (const token of new Set(tokens)) {
      const idxRes = await kv.get<{ docIds: string[] }>(["index", token]);
      const docIds = idxRes.value?.docIds || [];
      if (!docIds.includes(docId)) {
        docIds.push(docId);
        await kv.set(["index", token], { docIds });
      }
    }

    await kv.set(["documents", docId], { text: text.substring(0, 50000), tf: tfNorm, N });
  },

  async search(query: string, topK = 3): Promise<{ docId: string; score: number; snippet: string }[]> {
    const qTokens = await this.tokenize(query);
    const qTf: Record<string, number> = {};
    qTokens.forEach((t) => { qTf[t] = (qTf[t] || 0) + 1; });
    
    // ИСПРАВЛЕНИЕ 1 (применено снова)
    const qTfValues = Object.values(qTf);
    const maxQTf = qTfValues.length > 0 ? Math.max(...qTfValues) : 1;
    
    const metaRes = await kv.get<{ N: number }>(["meta", "rag"]);
    const N = metaRes.value?.N || 1;

    const qVector: Record<string, number> = {};
    for (const [token, count] of Object.entries(qTf)) {
      const idxRes = await kv.get<{ docIds: string[] }>(["index", token]);
      const df = idxRes.value?.docIds.length || 1;
      const idf = Math.log(N / df);
      qVector[token] = (count / maxQTf) * idf;
    }

    const candidateDocs = new Set<string>();
    for (const token of Object.keys(qVector)) {
      const idxRes = await kv.get<{ docIds: string[] }>(["index", token]);
      idxRes.value?.docIds.forEach((id) => candidateDocs.add(id));
    }

    const scores: { docId: string; score: number; snippet: string }[] = [];
    for (const docId of candidateDocs) {
      const docRes = await kv.get<{ text: string; tf: Record<string, number>; N: number }>(["documents", docId]);
      if (!docRes.value) continue;

      const { text, tf: docTf, N: docN } = docRes.value;
      let dotProduct = 0;
      let normQ = 0;
      let normD = 0;

      for (const [token, qVal] of Object.entries(qVector)) {
        const dVal = docTf[token] ? docTf[token] * Math.log(docN / (Object.keys(docTf).length || 1)) : 0;
        dotProduct += qVal * dVal;
        normQ += qVal ** 2;
        normD += dVal ** 2;
      }

      const cosineSim = (normQ > 0 && normD > 0) ? dotProduct / (Math.sqrt(normQ) * Math.sqrt(normD)) : 0;
      if (cosineSim > 0) {
        scores.push({
          docId,
          score: Number(cosineSim.toFixed(4)),
          snippet: text.substring(0, 150) + "...",
        });
      }
    }

    return scores.sort((a, b) => b.score - a.score).slice(0, topK);
  },
};

// --- МОДУЛЬ 3: ПАРСЕР ЛОГОВ ---
const LogParser = {
  pattern: /(?<ip>\d+\.\d+\.\d+\.\d+) - - \[(?<date>.*?)\] "(?<method>\w+) (?<path>.*?) HTTP\/.*?" (?<status>\d+) (?<size>\d+)/,
  
  parse(logText: string) {
    const lines = logText.trim().split("\n");
    const stats = { total: 0, statusCodes: {} as Record<string, number>, topIps: {} as Record<string, number>, topPaths: {} as Record<string, number> };

    for (const line of lines) {
      const match = line.match(this.pattern);
      if (match?.groups) {
        const { ip, status, path } = match.groups;
        stats.total++;
        stats.statusCodes[status] = (stats.statusCodes[status] || 0) + 1;
        stats.topIps[ip] = (stats.topIps[ip] || 0) + 1;
        stats.topPaths[path] = (stats.topPaths[path] || 0) + 1;
      }
    }

    // ИСПРАВЛЕНИЕ 2: Явные типы для sort и reduce, чтобы строгий TS не падал
    const sortDesc = (obj: Record<string, number>): Record<string, number> =>
      Object.entries(obj)
        .sort(([, a]: [string, number], [, b]: [string, number]) => b - a)
        .slice(0, 5)
        .reduce<Record<string, number>>((r, [k, v]) => ({ ...r, [k]: v }), {});

    return {
      entriesCount: stats.total,
      stats: {
        statusCodes: stats.statusCodes,
        topIps: sortDesc(stats.topIps),
        topPaths: sortDesc(stats.topPaths),
      },
    };
  },
};

// --- ОБРАБОТЧИКИ ЗАПРОСОВ (ROUTES) ---
async function handleRequest(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  const url = new URL(req.url);
  const path = url.pathname;
  const method = req.method;

  try {
    if (method === "GET" && path === "/api/health") {
      return jsonResponse({ status: "ok", runtime: "Deno Deploy" });
    }

    if (method === "POST" && path === "/api/licenses/activate") {
      const body = await req.json() as { key: string; device_fingerprint: string };
      const key = body.key.toUpperCase().trim();
      const demoKeys: Record<string, "pro" | "enterprise"> = { "NEXUS-PRO-2024-KEY": "pro", "NEXUS-ENT-2024-KEY": "enterprise" };
      
      let plan: "pro" | "enterprise" | "trial" = "trial";
      let days = 7;
      if (demoKeys[key]) { plan = demoKeys[key]; days = 30; }
      else if (/^[A-Z0-9]{5}-[A-Z0-9]{5}-[A-Z0-9]{5}-[A-Z0-9]{5}$/.test(key)) { plan = "pro"; days = 7; }
      else { return jsonResponse({ error: "INVALID_KEY_FORMAT" }, 400); }

      const expiresAt = new Date(Date.now() + days * 24 * 60 * 60 * 1000).toISOString();
      const license: License = { key, plan, expiresAt, deviceFingerprint: body.device_fingerprint, activatedAt: new Date().toISOString(), userId: null };
      
      await kv.set(["licenses", key], license);
      return jsonResponse({ plan: license.plan, key: license.key, expiresAt: license.expiresAt, activatedAt: license.activatedAt });
    }

    if (method === "GET" && path.startsWith("/api/license/status")) {
      const key = url.searchParams.get("key");
      if (!key) return jsonResponse({ valid: false, error: "No key provided" }, 400);
      const res = await kv.get<License>(["licenses", key]);
      if (!res.value || new Date(res.value.expiresAt) < new Date()) {
        return jsonResponse({ valid: false, error: "License not found or expired" });
      }
      return jsonResponse({ valid: true, plan: res.value.plan, expiresAt: res.value.expiresAt });
    }

    if (method === "POST" && path === "/api/accounts/selfemployed") {
      const body = await req.json() as AccountSE;
      const id = generateId();
      await kv.set(["accounts_se", id], { ...body, id });
      return jsonResponse({ status: "success", data: { ...body, id } });
    }
    if (method === "GET" && path === "/api/accounts/selfemployed") {
      const iter = kv.list<AccountSE>({ prefix: ["accounts_se"] }, { limit: 1, reverse: true });
      const latest = await iter.next();
      return jsonResponse({ data: latest.value || null });
    }

    if (method === "POST" && path === "/api/accounts/ip") {
      const body = await req.json() as AccountIP;
      const id = generateId();
      await kv.set(["accounts_ip", id], { ...body, id });
      return jsonResponse({ status: "success", data: { ...body, id } });
    }
    if (method === "GET" && path === "/api/accounts/ip") {
      const iter = kv.list<AccountIP>({ prefix: ["accounts_ip"] }, { limit: 1, reverse: true });
      const latest = await iter.next();
      return jsonResponse({ data: latest.value || null });
    }

    if (method === "DELETE" && path === "/api/accounts/all") {
      // ИСПРАВЛЕНИЕ 3: Явный тип <unknown> для kv.list при удалении
      for await (const entry of kv.list<unknown>({ prefix: ["accounts_se"] })) {
        await kv.delete(entry.key);
      }
      for await (const entry of kv.list<unknown>({ prefix: ["accounts_ip"] })) {
        await kv.delete(entry.key);
      }
      return jsonResponse({ status: "success", message: "All accounts purged" });
    }

    if (method === "POST" && path === "/api/missions") {
      const body = await req.json() as Omit<Mission, "id">;
      const id = generateId();
      await kv.set(["missions", id], { ...body, id });
      return jsonResponse({ status: "success", id });
    }

    if (method === "POST" && path === "/api/anonymize") {
      const { text } = await req.json() as { text: string };
      return jsonResponse({ original_length: text.length, anonymized: Anonymizer.anonymize(text) });
    }

    if (method === "POST" && path === "/api/index/build") {
      const { doc_id, text } = await req.json() as { doc_id: string; text: string };
      await RAGEngine.buildIndex(doc_id, text);
      return jsonResponse({ status: "success", doc_id });
    }

    if (method === "POST" && path === "/api/search") {
      const { query, top_k = 3 } = await req.json() as { query: string; top_k?: number };
      return jsonResponse({ results: await RAGEngine.search(query, top_k) });
    }

    if (method === "POST" && path === "/api/logs/parse") {
      const { log_text } = await req.json() as { log_text: string };
      return jsonResponse(LogParser.parse(log_text));
    }

    if (method === "POST" && path === "/api/logs/analyze") {
      const { log_text } = await req.json() as { log_text: string };
      const parsed = LogParser.parse(log_text);
      const total = parsed.entriesCount;
      const errors = Object.entries(parsed.stats.statusCodes)
        .filter(([code]) => code.startsWith("4") || code.startsWith("5"))
        .reduce((sum, [, count]) => sum + (count as number), 0);
      const errorRate = total > 0 ? (errors / total) * 100 : 0;

      const artifacts: { type: string; title: string; content: string }[] = [];
      if (errorRate > 10) {
        artifacts.push({ type: "anomaly", title: "High Error Rate", content: `Error rate is ${errorRate.toFixed(2)}%` });
      }
      
      return jsonResponse({ parsed, artifacts, errorRate: Number(errorRate.toFixed(2)) });
    }

    if (method === "POST" && path === "/api/payments/create") {
      const body = await req.json() as { plan: string; amount: number | string; payment_method: string };
      const amount = typeof body.amount === "string" ? parseFloat(body.amount) : (body.amount || (body.plan === "pro" ? 4999 : 24999));
      const paymentId = `pay_${crypto.randomUUID().replace(/-/g, "").substring(0, 16)}`;
      
      await kv.set(["payments", paymentId], { paymentId, amount, plan: body.plan, status: "succeeded", createdAt: new Date().toISOString() });

      const days = body.plan === "enterprise" ? 365 : 30;
      const licenseKey = `NEXUS-${body.plan.toUpperCase()}-${crypto.randomUUID().replace(/-/g, "").substring(0, 8).toUpperCase()}`;
      const expiresAt = new Date(Date.now() + days * 24 * 60 * 60 * 1000).toISOString();
      
      await kv.set(["licenses", licenseKey], { 
        key: licenseKey, plan: body.plan, expiresAt, deviceFingerprint: null, activatedAt: new Date().toISOString(), userId: null 
      });

      return jsonResponse({ 
        confirmed: true, 
        payment_id: paymentId, 
        license: { key: licenseKey, expires_at: expiresAt } 
      });
    }

    if (method === "GET" && path.startsWith("/api/export/")) {
      const type = path.split("/").pop();
      return jsonResponse({ type, timestamp: new Date().toISOString(), data: "mock_exported_data_from_deno_kv" });
    }

    if (method === "POST" && path === "/api/documents/upload") {
      const formData = await req.formData();
      const file = formData.get("file");
      const docId = (formData.get("doc_id") as string) || generateId();
      
      if (!file || !(file instanceof File)) {
        return jsonResponse({ error: "No valid file provided" }, 400);
      }
      
      const text = await file.text();
      if (text.length > 50000) {
        return jsonResponse({ error: "File too large for Deno KV (max ~50KB). Use external storage for larger files." }, 413);
      }
      
      await RAGEngine.buildIndex(docId, text);
      return jsonResponse({ status: "success", doc_id: docId, size: text.length });
    }

    return jsonResponse({ error: "Not Found", path }, 404);

  } catch (err) {
    console.error("API Error:", err);
    return jsonResponse({ error: "Internal Server Error", detail: String(err) }, 500);
  }
}

// --- ЗАПУСК СЕРВЕРА ---
Deno.serve((req) => {
  return handleRequest(req);
});
