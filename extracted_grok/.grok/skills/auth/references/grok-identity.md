# Sign in with Grok (deployed apps — zero clicks)

Behind the edge gate, every proxied request from a signed-in Grok viewer
carries an unforgeable `x-grok-identity` JWT (EdDSA, minted per request; the
gate strips any client-supplied copy). The pre-wired `gateIdentitySessions`
plugin (`src/lib/auth/gate-session.server.ts`) verifies it against the gate's
JWKS (`/__gate/identity-key`, via `src/lib/auth/gate-identity.server.ts`) and,
when the app has no session yet, materializes the Better Auth session for that
viewer automatically — no sign-in button, no redirect, no broker round-trip.
`useSession` / `useCurrentUser` simply return the Grok user.

The broker OAuth flow is the **fallback** for anonymous/public viewers and for
contexts without the gate; the live preview keeps its existing popup mechanism.

## Connector / app-data apps: gate sign-in only

When the app calls connector tools (the `app-data` skill applies), the login
page offers ONLY "Continue with Grok": the zero-click `x-grok-identity` session
above, or the gate-built `loginUrl` returned by a connector call. Do not wire
Google/X buttons for these apps — a broker login can mint an identity that is
not the gate viewer the connector data belongs to. The three-method rule
applies to apps without connector data. Still no new `GROK_PROVIDERS` entries,
still never edit `src/lib/auth/`.

## Files (pre-wired — do not edit)

| File | Role |
|---|---|
| `gate-identity.server.ts` | Verifies the gate's `x-grok-identity` viewer JWT (EdDSA vs the gate JWKS; fail-closed). Server-only. |
| `gate-session.server.ts` | Better Auth plugin that turns a verified gate identity into the app session with zero clicks. Already registered in `server.ts`. |

## Env (deployer-injected)

| Var | Scope | Meaning |
|---|---|---|
| `GROK_PROJECT_ID` | server | enables "Sign in with Grok" (`x-grok-identity` audience check `app:<project_id>`) |
| `GROK_GATE_ORIGIN` | server | gate public origin (JWKS + issuer pin); unset → derived from the inbound host |

Deployed behavior: gate-authenticated viewers are signed in automatically from
`x-grok-identity`; the deployer also injects a per-app broker client +
`DATABASE_URL`, so the fallback sign-in persists identities in Postgres.

## Connector / app-data apps: gate sign-in only

When the `app-data` skill applies, the login page offers ONLY "Continue with
Grok" via the gate sign-in — the zero-click `x-grok-identity` session above, or
the gate-built `loginUrl` returned by a connector call. Do not wire Google/X
buttons for these apps: a broker login can mint an identity that is not the
gate viewer the connector data belongs to. The three-method rule applies to
apps without connector data. Still no new `GROK_PROVIDERS` entries, still never
edit `src/lib/auth/`.
