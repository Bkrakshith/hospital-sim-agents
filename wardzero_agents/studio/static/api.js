/* Thin same-origin client for the Studio API. Writes carry the header the server requires. */
(function () {
  class ApiError extends Error {
    constructor(status, message) { super(message); this.status = status; }
  }

  async function req(method, path, body) {
    const init = { method, headers: {} };
    if (method !== "GET") init.headers["X-WardZero-Panel"] = "1";
    if (body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    let res;
    try { res = await fetch(path, init); }
    catch (e) { throw new ApiError(0, "The Studio server cannot be reached."); }
    let data = null;
    try { data = await res.json(); } catch (e) { /* empty body */ }
    if (!res.ok) throw new ApiError(res.status, (data && (data.error || data.message)) || ("Request failed (" + res.status + ")."));
    return data;
  }

  /* Live run state from /ws, reconnecting quietly. */
  function watchState(onState) {
    let ws = null, delay = 500, dead = false;
    function open() {
      if (dead) return;
      const proto = location.protocol === "https:" ? "wss://" : "ws://";
      try { ws = new WebSocket(proto + location.host + "/ws"); } catch (e) { return retry(); }
      ws.onmessage = (ev) => { try { onState(JSON.parse(ev.data)); } catch (e) { /* ignore */ } };
      ws.onopen = () => { delay = 500; };
      ws.onclose = retry;
    }
    function retry() { if (!dead) { setTimeout(open, delay); delay = Math.min(delay * 2, 8000); } }
    open();
    return () => { dead = true; if (ws) ws.close(); };
  }

  window.API = {
    ApiError,
    get: (p) => req("GET", p),
    post: (p, b) => req("POST", p, b === undefined ? {} : b),
    put: (p, b) => req("PUT", p, b),
    del: (p) => req("DELETE", p),
    watchState,
  };
})();
