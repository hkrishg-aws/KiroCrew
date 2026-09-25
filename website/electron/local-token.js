"use strict";

/**
 * The literal-loopback origin of `backendUrl` and the port it addresses.
 *
 * One parse serves both, because the credential is a function of the dial
 * target: resolving the origin here and the port somewhere else would let a
 * caller authenticate for one listener while dialing another.
 *
 * `URL.port` is "" on a scheme-default port, so the default is spelled out:
 * an empty port would name a credential file no gateway ever wrote.
 *
 * @param {string} backendUrl
 * @returns {{origin: string, port: string} | null} null when the URL is not a
 *   literal `http:` loopback origin, which is the only target a local secret
 *   may be sent to.
 */
function loopbackTarget(backendUrl) {
  try {
    const url = new URL(backendUrl);
    if (url.protocol !== "http:") return null;
    if (url.hostname === "localhost" || url.hostname === "kirocrew.localhost") {
      url.hostname = "127.0.0.1";
    }
    if (url.hostname !== "127.0.0.1") return null;
    return { origin: url.origin, port: url.port || "80" };
  } catch {
    return null;
  }
}

function literalLoopbackUrl(backendUrl) {
  const target = loopbackTarget(backendUrl);
  return target ? target.origin : "";
}

/**
 * Bind addresses whose listener necessarily answers the dialed v4 loopback.
 *
 * The dialed address is always `127.0.0.1` -- `loopbackTarget` accepts nothing
 * else -- so these are the only two spellings under which a published credential
 * can belong to the party actually reached. A v4 loopback bind IS that party; a
 * v4 wildcard bind covers it.
 *
 * Every other address is refused, `::` and `::1` included: a v6 bind leaves IPv4
 * `127.0.0.1:<port>` unbound and free for a co-resident to take, so its
 * credential belongs to a listener this call never spoke to. Whether a v6
 * wildcard also accepts v4-mapped connections depends on the host's
 * `IPV6_V6ONLY` setting, which is not a fact this side can establish, so it is
 * treated as a different listener. The same family reasoning governs the
 * gateway's own callback-host export.
 *
 * Trying both is NOT a fallback between listeners: each names a listener on the
 * address being dialed, so neither can resolve to a party that was not reached.
 */
const LOOPBACK_COVERING_BINDS = ["127.0.0.1", "0.0.0.0"];

/**
 * Path of the gateway credential for the listener at `bindAddress` on `port`.
 *
 * A gateway publishes its own in-memory credential, once it has bound, as
 * `run/gateway-<port>-<address>.secret`: mode 0600 inside an owner-only `run/`
 * directory. Reading that file answers "may a secret go to whatever answers this
 * address and port?" out of local disk state, so nothing is asked of the peer --
 * a process that merely holds the port, whatever it presents itself as, is never
 * consulted and never believed.
 *
 * The name carries the address because a port number identifies a SET of
 * listeners. Keyed by port alone, a gateway bound to the v6 loopback and a
 * tunnel's local end on v4 share one entry, and the app would read the former's
 * credential and send it to the latter.
 *
 * @param {string} home data home whose `config.json` governs this launch
 * @param {string} port port being dialed
 * @param {string} bindAddress address whose listener published the credential
 * @param {object} path node:path (injected)
 * @returns {string}
 */
function listenerSecretPath(home, port, bindAddress, path) {
  return path.join(home, "run", `gateway-${port}-${bindAddress}.secret`);
}

async function requestLocalToken(http, literalUrl, secret) {
  if (!literalUrl) return "";
  return new Promise((resolve) => {
    const req = http.get(
      `${literalUrl}/api/token/local`,
      { headers: { "X-Local-Secret": secret }, timeout: 5000 },
      (res) => {
        if (res.statusCode !== 200) {
          res.resume();
          resolve("");
          return;
        }
        let data = "";
        res.on("error", () => resolve(""));
        res.on("data", (chunk) => { data += chunk; });
        res.on("end", () => {
          try { resolve(JSON.parse(data).token || ""); } catch { resolve(""); }
        });
      },
    );
    req.on("error", () => resolve(""));
    req.on("timeout", () => { req.destroy(); resolve(""); });
  });
}

/**
 * A dashboard token minted against the gateway that owns the dialed listener.
 *
 * The credential sent is the one that gateway published for its own listener,
 * identified by the address AND the port that were dialed. Neither the home-wide
 * `.local_secret` nor a port-keyed entry is a second place to look. The shared
 * file holds one slot per data home on a last-writer-wins basis, and a port-keyed
 * entry names every listener sharing that port number, so either can hold a
 * credential belonging to a gateway this call never reached. Sending that to
 * whoever answers here would surrender a credential which authenticates
 * elsewhere. An entry keyed by the dialed listener cannot escalate, because the
 * only listener it authenticates against is the one it was just sent to.
 *
 * An absent entry therefore denies rather than widens. A port no local gateway
 * bound on this address, an `ssh -L` forward's local end among them, whether or
 * not a v6-bound gateway holds the same port number, has no credential to read,
 * so the caller falls through to the remote-token path and then to the token
 * prompt.
 *
 * A REFUSED entry is not the end of the walk either. `clear_marker` runs only on
 * a graceful shutdown, so a crashed gateway leaves its entry behind, and a stale
 * exact-address entry sitting beside the live wildcard one would otherwise spend
 * the single attempt and report no token while a working credential was never
 * tried. Continuing costs nothing that matters: a dead generation's secret
 * authenticates against no listener at all, and every candidate names a listener
 * on the address being dialed, so the walk never reaches a party this call did
 * not reach.
 */
async function fetchLocalToken({ backendUrl, resolveHome, path, fs, http }) {
  const target = loopbackTarget(backendUrl);
  if (!target) return "";
  const authoritativeHome = resolveHome();
  for (const bindAddress of LOOPBACK_COVERING_BINDS) {
    let secret = "";
    try {
      secret = fs
        .readFileSync(listenerSecretPath(authoritativeHome, target.port, bindAddress, path), "utf8")
        .trim();
    } catch {
      // A missing/unreadable entry is an ordinary token miss.
    }
    if (!secret) continue;
    const token = await requestLocalToken(http, target.origin, secret);
    if (token) return token;
  }
  return "";
}

module.exports = {
  fetchLocalToken,
  literalLoopbackUrl,
  listenerSecretPath,
  LOOPBACK_COVERING_BINDS,
};
