// ============================================================================
// NEXUS MARKETING AGENTS v5.0 // DENO DEPLOY + ROBOKASSA INTEGRATION
// ============================================================================

// 1. УМНАЯ ИНИЦИАЛИЗАЦИЯ KV (с защитой от падения сборки в Preview)
let kvInstance: Deno.Kv | null = null;

async function getKv(): Promise<Deno.Kv> {
  if (!kvInstance) {
    try {
      kvInstance = await Deno.openKv();
    } catch (e) {
      console.warn("⚠️ Deno KV не подключен в этой среде (Preview). Используется временная память.");
      kvInstance = createMockKv() as unknown as Deno.Kv;
    }
  }
  return kvInstance;
}

function createMockKv() {
  const store = new Map<string, any>();
  return {
    get: async (key: unknown[]) => ({ value: store.get(JSON.stringify(key)) || null, versionstamp: "mock" }),
    set: async (key: unknown[], value: unknown) => { store.set(JSON.stringify(key), value); return { versionstamp: "mock" }; },
    delete: async (key: unknown[]) => { store.delete(JSON.stringify(key)); },
    list: async function* (selector: { prefix?: unknown[]; limit?: number; reverse?: boolean }) {
      const prefixStr = selector.prefix ? JSON.stringify(selector.prefix) : "";
      let entries = Array.from(store.entries()).filter(([k]) => k.startsWith(prefixStr));
      if (selector.reverse) entries.reverse();
      if (selector.limit) entries = entries.slice(0, selector.limit);
      for (const [k, v] of entries) {
        yield { key: JSON.parse(k), value: v, versionstamp: "mock" };
      }
    },
    close: () => {},
    atomic: () => ({ check: () => ({}), set: () => ({}), delete: () => ({}), sum: () => ({}), min: () => ({}), max: () => ({}), commit: async () => ({ ok: true, versionstamp: "mock" }) }),
    listenQueue: () => {},
    enqueue: async () => ({ ok: true })
  };
}

const kv = await getKv();

// ============================================================================
// 2. ТИПЫ И УТИЛИТЫ
// ============================================================================
interface License { key: string; plan: string; expiresAt: string; deviceFingerprint: string | null; activatedAt: string; userId: string | null; }
interface AccountSE { id: string; fio: string; inn: string; email: string; phone: string; regDate: string; }
interface AccountIP { id: string; name: string; inn: string; ogrnip: string; email: string; phone: string; account: string; bank: string; bik: string; address: string; }

const corsHeaders = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS", "Access-Control-Allow-Headers": "Content-Type, Authorization" };
const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json", ...corsHeaders } });
const uuid = () => crypto.randomUUID();

// ============================================================================
// 3. MD5 (Требуется Робокассой для подписи)
// ============================================================================
async function md5(message: string): Promise<string> {
  const msgBuffer = new TextEncoder().encode(message);
  const hashBuffer = await crypto.subtle.digest('MD5', msgBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

// ============================================================================
// 4. МОДУЛИ (Анонимизация, RAG)
// ============================================================================
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

// ============================================================================
// 5. ОБРАБОТЧИК ЗАПРОСОВ
// ============================================================================
async function handleRequest(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });
  const url = new URL(req.url);
  const path = url.pathname;

  try {
    // Health Check
    if (req.method === "GET" && path === "/api/health") {
      return json({ status: "ok", runtime: "Deno Deploy", kv_mode: kvInstance ? "Real" : "Mock", robokassa: "integrated" });
    }

    // === ЛИЦЕНЗИИ ===
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

    // === АККАУНТЫ ===
    if (req.method === "POST" && path === "/api/accounts/selfemployed") {
      const body = await req.json() as AccountSE;
      const id = uuid();
      await kv.set(["accounts_se", id], { ...body, id });
      return json({ status: "success", data: { ...body, id } });
    }
    
    if (req.method === "GET" && path === "/api/accounts/selfemployed") {
      const iter = kv.list<AccountSE>({ prefix: ["accounts_se"] }, { limit: 1, reverse: true });
      const items = [];
      for await (const entry of iter) items.push(entry.value);
      return json({ data: items[0] || null });
    }

    if (req.method === "POST" && path === "/api/accounts/ip") {
      const body = await req.json() as AccountIP;
      const id = uuid();
      await kv.set(["accounts_ip", id], { ...body, id });
      return json({ status: "success", data: { ...body, id } });
    }

    if (req.method === "GET" && path === "/api/accounts/ip") {
      const iter = kv.list<AccountIP>({ prefix: ["accounts_ip"] }, { limit: 1, reverse: true });
      const items = [];
      for await (const entry of iter) items.push(entry.value);
      return json({ data: items[0] || null });
    }

    if (req.method === "DELETE" && path === "/api/accounts/all") {
      for await (const entry of kv.list({ prefix: ["accounts_se"] })) await kv.delete(entry.key);
      for await (const entry of kv.list({ prefix: ["accounts_ip"] })) await kv.delete(entry.key);
      return json({ status: "success", message: "Purged" });
    }

    // === МИССИИ ===
    if (req.method === "POST" && path === "/api/missions") {
      const body = await req.json();
      const id = uuid();
      await kv.set(["missions", id], { ...body, id });
      return json({ status: "success", id });
    }

    // === АНОНИМИЗАЦИЯ ===
    if (req.method === "POST" && path === "/api/anonymize") {
      const { text } = await req.json() as { text: string };
      return json({ original_length: text.length, anonymized: Anonymizer.anonymize(text) });
    }

    // === RAG ===
    if (req.method === "POST" && path === "/api/index/build") {
      const { doc_id, text } = await req.json() as { doc_id: string; text: string };
      await RAGEngine.buildIndex(doc_id, text);
      return json({ status: "success", doc_id });
    }

    if (req.method === "POST" && path === "/api/search") {
      const { query, top_k = 3 } = await req.json() as { query: string; top_k?: number };
      return json({ results: await RAGEngine.search(query, top_k) });
    }

    // === ЛОГИ ===
    if (req.method === "POST" && path === "/api/logs/parse") {
      const { log_text } = await req.json() as { log_text: string };
      return json({ parsed: "mock_parsed_data", entriesCount: log_text.split("\n").length });
    }

    // ====================================================================
    // === РОБОКАССА: СОЗДАНИЕ ПЛАТЕЖА ===
    // ====================================================================
    if (req.method === "POST" && path === "/api/payments/create") {
      const body = await req.json() as { user_name: string; amount: number; inv_id: string; email?: string };
      
      const login = Deno.env.get("ROBOKASSA_LOGIN") || "your_login";
      const pass1 = Deno.env.get("ROBOKASSA_PASSWORD_1") || "your_password1";
      const amount = body.amount || 990;
      const invId = body.inv_id || Date.now().toString();
      const description = "Подписка NEXUS MARKETING AGENTS на 1 месяц";
      const email = body.email || "";

      // Формируем подпись: Login:OutSum:InvId:Password1
      const signatureString = `${login}:${amount}:${invId}:${pass1}`;
      const signature = await md5(signatureString);

      // Определяем тестовый или боевой режим
      const isTest = login === "your_login" || login === "test_login";
      const baseUrl = "https://auth.robokassa.ru/Merchant/Index.aspx";
      
      // Формируем URL оплаты
      let paymentUrl = `${baseUrl}?MerchantLogin=${encodeURIComponent(login)}&OutSum=${amount}&InvoiceID=${encodeURIComponent(invId)}&Description=${encodeURIComponent(description)}&SignatureValue=${signature}`;
      
      if (email) paymentUrl += `&Email=${encodeURIComponent(email)}`;
      if (isTest) paymentUrl += `&IsTest=1`;

      console.log(`[PAYMENT] Created invoice ${invId} for ${amount} RUB (user: ${body.user_name})`);

      return json({ 
        success: true, 
        payment_url: paymentUrl, 
        inv_id: invId,
        amount: amount,
        is_test: isTest
      });
    }

    // ====================================================================
    // === РОБОКАССА: ВЕБХУК (Result URL) ===
    // ====================================================================
    if (req.method === "POST" && path === "/api/payments/robokassa-result") {
      const formData = await req.formData();
      const outSum = (formData.get("OutSum") as string) || "";
      const invId = (formData.get("InvoiceID") as string) || "";
      const receivedSignature = (formData.get("SignatureValue") as string) || "";
      const userEmail = (formData.get("Email") as string) || "unknown";

      const pass2 = Deno.env.get("ROBOKASSA_PASSWORD_2") || "your_password2";
      
      // Проверяем подпись: OutSum:InvoiceID:Password2
      const checkString = `${outSum}:${invId}:${pass2}`;
      const expectedSignature = await md5(checkString);

      if (receivedSignature.toUpperCase() !== expectedSignature.toUpperCase()) {
        console.error(`[PAYMENT] Bad signature for invoice ${invId}`);
        return new Response("bad sign", { status: 400 });
      }

      console.log(`[PAYMENT] ✅ Payment confirmed! Invoice: ${invId}, Sum: ${outSum}, Email: ${userEmail}`);

      // Подпись верна — начисляем подписку
      const licenseKey = `NEXUS-PRO-${crypto.randomUUID().replace(/-/g, "").substring(0, 8).toUpperCase()}`;
      const expiresAt = new Date(Date.now() + 30 * 86400000).toISOString();
      
      await kv.set(["licenses", licenseKey], { 
        key: licenseKey, 
        plan: "pro", 
        expiresAt, 
        deviceFingerprint: userEmail, 
        activatedAt: new Date().toISOString(), 
        userId: null 
      });

      await kv.set(["payments", invId], { 
        invId, 
        amount: parseFloat(outSum), 
        plan: "pro", 
        status: "succeeded", 
        email: userEmail,
        date: new Date().toISOString() 
      });

      // Робокасса требует строго такой ответ
      return new Response(`OK${invId}`, { status: 200 });
    }

    // ====================================================================
    // === РОБОКАССА: SUCCESS URL (редирект после оплаты) ===
    // ====================================================================
    if (req.method === "GET" && path === "/api/payments/success") {
      const outSum = url.searchParams.get("OutSum") || "";
      const invId = url.searchParams.get("InvoiceID") || "";
      return json({ 
        status: "success", 
        message: "Оплата успешно завершена", 
        inv_id: invId, 
        amount: outSum,
        redirect: "https://n49149022-lgtm.github.io/NEXUS/?payment=success"
      });
    }

    // === ЭКСПОРТ ===
    if (req.method === "GET" && path.startsWith("/api/export/")) {
      return json({ type: path.split("/").pop(), timestamp: new Date().toISOString(), data: "mock_export" });
    }

    // === ЗАГРУЗКА ФАЙЛОВ ===
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

// Запуск сервера
Deno.serve(handleRequest);
