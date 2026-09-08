# Cloud Deployment — AuraDB + Azure

End-to-end runbook to move GraphRAG Explorer from a local Docker stack to:

```
Static Web Apps (React)  ──►  Container Apps (FastAPI)  ──►  Neo4j AuraDB
       │                             │
   GitHub Actions              Key Vault (NEO4J_*, GOOGLE_API_KEY, GITHUB_TOKEN)
       └──────── build image → push ACR → update Container App ────────┘
```

Local Docker (`docker compose`) keeps working unchanged — the only switch is `NEO4J_URI`.

Placeholders: `<region>` (e.g. `centralindia`), `<acr>` (globally-unique, lowercase),
`<kv>` (globally-unique), `<sub-id>`, `<repo>` = `owner/name` of this GitHub repo.

Prereqs: `az` CLI (`az login`), `docker`, a GitHub repo for this project, a Neo4j Aura account.

---

## Part D — AuraDB instance + data migration

### D1. Create the instance

1. <https://console.neo4j.io> → **New Instance** → **AuraDB Free** → Neo4j 5.x.
2. Download / copy the generated password. Note the connection URI:
   `neo4j+s://<id>.databases.neo4j.io`.

### D2. Confirm the graph fits Free tier (200k nodes / 400k relationships)

Against the local graph:

```bash
docker exec neo4j-graphrag cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "MATCH (n) RETURN count(n) AS nodes"
docker exec neo4j-graphrag cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "MATCH ()-[r]->() RETURN count(r) AS rels"
```

**Measured 2026-09-08: 20,447 nodes / 173,467 relationships — well within Free tier.**
6 repos ingested; 5 vector indexes (`code_/module_/type_/var_/issue_embeddings`).

Over the cap → use **AuraDB Professional**, or ingest a subset. **Stop here and decide before D3.**

### D3. Dump the local graph  ✅ done — `dump/neo4j.dump` (705 MB, git-ignored)

The live data is in the `neo4j-graphrag` container's **anonymous** volume
(`5c78808172ff…`), not the compose `graphrag_neo4j_data` volume. Find it with:

```bash
VOL=$(docker inspect neo4j-graphrag \
  -f '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}')
```

To (re)create the dump:

```bash
mkdir -p dump && chmod 777 dump
docker stop neo4j-graphrag
docker run --rm --user root -v "$VOL":/data -v "$PWD/dump":/dump neo4j:5.20 \
  neo4j-admin database dump neo4j --to-path=/dump --overwrite-destination=true
docker start neo4j-graphrag
```

> Running a second `neo4j:5.20` container at the same time can trip Docker's OOM
> killer (each wants ~1 GB heap). Stop one, or raise Docker Desktop's memory.

### D4. Upload the dump to AuraDB  ✅ done 2026-09-08 → instance `5e70640f`

> **Use `neo4j:5.26`, not `neo4j:5.20`.** The 5.20 `neo4j-admin database upload`
> auth handshake fails against the current Aura API with
> `Unexpected response code 411 from request: Authorization`. 5.26 (5.x LTS) works.

> **Newer Aura credentials files** (`Neo4j-<id>-Created-*.txt`) set
> `NEO4J_USERNAME` / `NEO4J_DATABASE` to the **instance id**, not `neo4j`.
> Pass those through — the upload reads `NEO4J_USERNAME` / `NEO4J_PASSWORD` from env.

```bash
CF=~/Downloads/Neo4j-<id>-Created-<date>.txt
export NEO4J_USERNAME=$(grep '^NEO4J_USERNAME=' "$CF" | cut -d= -f2-)
export NEO4J_PASSWORD=$(grep '^NEO4J_PASSWORD=' "$CF" | cut -d= -f2-)
A_URI=$(grep '^NEO4J_URI=' "$CF" | cut -d= -f2-)

docker run --rm -e NEO4J_USERNAME -e NEO4J_PASSWORD -v "$PWD/dump":/dump \
  --entrypoint neo4j-admin neo4j:5.26 \
  database upload neo4j --from-path=/dump --to-uri="$A_URI" --overwrite-destination=true
```

Takes ~1 min to upload + a few min for Aura's server-side import. On success it
prints `Your data was successfully pushed to Aura and is now running.` and the
local `dump/neo4j.dump` can be deleted (kept here as a backup; it's git-ignored).

### D5. Verify on AuraDB  ✅ exact parity

```bash
q(){ docker run --rm -e NEO4J_PASSWORD --entrypoint cypher-shell neo4j:5.26 \
       -a "$A_URI" -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" --format plain "$1"; }
q "MATCH (n) RETURN count(n);"
q "MATCH ()-[r]->() RETURN count(r);"
q "SHOW INDEXES YIELD name,type,state WHERE type='VECTOR' RETURN name,state;"
q "MATCH (f:Function) RETURN count(f.embedding) AS with_embedding;"
```

Result 2026-09-08: **20,447 nodes / 173,467 rels** (matches D2 exactly), 6 repos,
5 vector indexes ONLINE (dim 3072), 12,659/12,659 Function embeddings intact, live
`db.index.vector.queryNodes` query returns hits. If vector indexes are ever missing,
recreate from `ingest/queries.py` (~lines 329–373).

### D6. Ingesting more repos later

No code change — point the env at Aura and run the pipeline as usual:

```bash
export NEO4J_URI=neo4j+s://<id>.databases.neo4j.io
export NEO4J_USER=neo4j
export NEO4J_PASSWORD='<aura-password>'
python backend_ingest.py           # TARGET_REPO from .env
```

---

## Part E — Azure resources

> **"Azure for Students" constraints hit during this deploy** (all worked around below):
> 1. **Allowed regions only**: `malaysiawest, indonesiacentral, koreacentral, uaenorth, eastasia`
>    (policy `sys.regionrestriction`).
> 2. **Gemini API geo-block**: the free Gemini API refuses requests from **Hong Kong
>    (`eastasia`)** → `FAILED_PRECONDITION: User location is not supported`. So the
>    **backend Container App runs in `koreacentral`**. Static Web Apps is region-locked to
>    a small list whose only overlap with the allowed regions is `eastasia`, so the
>    **frontend SWA stays in `eastasia`** (it's just static hosting — never calls Gemini).
>    ACR + Key Vault also stay in `eastasia`; cross-region ACR pull / KV read is fine.
> 3. **ACR Tasks disabled** (`az acr build` → `TasksOperationsNotAllowed`). → build locally / on
>    the CI runner with `docker build` + `docker push`.
> 4. **Container Apps env rejects `keyvaultref:` secrets** on this subscription. → Key Vault
>    still holds the canonical secrets; their **values are read from Key Vault and set as
>    inline Container App secrets** at deploy time.
> 5. **Tenant blocks app registrations** (`az ad app create` → Insufficient privileges). → GitHub
>    OIDC federates to the **user-assigned managed identity** instead of a service principal.
>
> **Region layout:** compute (`cae-graphrag`, `ca-graphrag-api`) in **`koreacentral`**;
> `graphragacr*`, `gr-kv-*`, `swa-graphrag` in **`eastasia`**; RG `rg-graphrag` metadata in `eastasia`.

Names used here: `rg-graphrag` · `graphragacr<suffix>` · `gr-kv-<suffix>` · `cae-graphrag`
(workload-profiles) · `id-graphrag` · `ca-graphrag-api` · `swa-graphrag`. The live values are
in the git-ignored `.azure-deploy.env` at the repo root.

### E1. Resource group + ACR

```bash
LOC=eastasia
az group create -n rg-graphrag -l $LOC
az acr create -n <acr> -g rg-graphrag -l $LOC --sku Basic
```

### E2. Key Vault + secrets

```bash
az keyvault create -n <kv> -g rg-graphrag -l $LOC --enable-rbac-authorization false

az keyvault secret set --vault-name <kv> -n neo4j-uri      --value "neo4j+s://<id>.databases.neo4j.io"
az keyvault secret set --vault-name <kv> -n neo4j-user     --value "<id>"          # instance id (newer Aura)
az keyvault secret set --vault-name <kv> -n neo4j-password --value "<aura-password>"
az keyvault secret set --vault-name <kv> -n google-api-key --value "<gemini-key>"
az keyvault secret set --vault-name <kv> -n github-token   --value "<github-pat>"   # ingest jobs only
```

### E3. Container Apps environment + managed identity

```bash
# compute goes in koreacentral (Gemini geo-block on eastasia)
az containerapp env create -n cae-graphrag -g rg-graphrag -l koreacentral \
  --enable-workload-profiles true --logs-destination none

az identity create -n id-graphrag -g rg-graphrag -l $LOC
MI_ID=$(az identity show -n id-graphrag -g rg-graphrag --query id -o tsv)
MI_PRINCIPAL=$(az identity show -n id-graphrag -g rg-graphrag --query principalId -o tsv)

# MI can read Key Vault secrets (used at deploy time) and pull from ACR
az keyvault set-policy -n <kv> --object-id "$MI_PRINCIPAL" --secret-permissions get list
az role assignment create --assignee-object-id "$MI_PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role AcrPull --scope $(az acr show -n <acr> --query id -o tsv)
```

### E4. Build & push the API image (local — ACR Tasks disabled)

```bash
az acr login -n <acr>
docker build --platform linux/amd64 -f backend/Dockerfile --target api \
  -t <acr>.azurecr.io/graphrag-api:bootstrap -t <acr>.azurecr.io/graphrag-api:latest .
docker push <acr>.azurecr.io/graphrag-api:bootstrap
docker push <acr>.azurecr.io/graphrag-api:latest
```

`--platform linux/amd64` matters on Apple Silicon — Container Apps runs amd64.

### E5. Create the backend Container App (inline secrets sourced from Key Vault)

Lands in `koreacentral` (follows its env). Note `--workload-profile-name Consumption`.

```bash
kv(){ az keyvault secret show --vault-name <kv> -n "$1" --query value -o tsv; }

az containerapp create \
  -n ca-graphrag-api -g rg-graphrag --environment cae-graphrag \
  --image <acr>.azurecr.io/graphrag-api:bootstrap \
  --registry-server <acr>.azurecr.io --registry-identity "$MI_ID" \
  --user-assigned "$MI_ID" --workload-profile-name Consumption \
  --ingress external --target-port 8000 \
  --min-replicas 1 --max-replicas 3 --cpu 0.5 --memory 1.0Gi \
  --secrets \
    "neo4j-uri=$(kv neo4j-uri)" \
    "neo4j-user=$(kv neo4j-user)" \
    "neo4j-password=$(kv neo4j-password)" \
    "google-api-key=$(kv google-api-key)" \
  --env-vars \
    NEO4J_URI=secretref:neo4j-uri \
    NEO4J_USER=secretref:neo4j-user \
    NEO4J_PASSWORD=secretref:neo4j-password \
    GOOGLE_API_KEY=secretref:google-api-key \
    ALLOWED_ORIGINS=https://placeholder.local \
    LOG_LEVEL=INFO

BACKEND_FQDN=$(az containerapp show -n ca-graphrag-api -g rg-graphrag \
  --query properties.configuration.ingress.fqdn -o tsv)
curl -s "https://$BACKEND_FQDN/api/health"   # {"status":"ok"}
curl -s "https://$BACKEND_FQDN/api/config"   # {"neo4j_preconfigured":true}
curl -s "https://$BACKEND_FQDN/api/stats"    # full graph from AuraDB
```

### E6. Static Web App for the frontend

Created **without** GitHub linkage (avoids the interactive OAuth flow); CI uses the
deployment token.

```bash
az staticwebapp create -n swa-graphrag -g rg-graphrag -l $LOC --sku Free
az staticwebapp secrets list -n swa-graphrag -g rg-graphrag --query properties.apiKey -o tsv  # → .swa-token
SWA_HOST=$(az staticwebapp show -n swa-graphrag -g rg-graphrag --query defaultHostname -o tsv)
```

The build is driven by [`.github/workflows/azure-static-web-apps.yml`](../../.github/workflows/azure-static-web-apps.yml)
(explicit `npm ci && npm run build` with `VITE_API_BASE`, then `Azure/static-web-apps-deploy@v1`
with `skip_app_build: true`). SPA routing config: `frontend/public/staticwebapp.config.json`.

### E7. Re-point CORS

```bash
az containerapp update -n ca-graphrag-api -g rg-graphrag \
  --set-env-vars ALLOWED_ORIGINS="https://$SWA_HOST"
```

Open `https://$SWA_HOST` — connects automatically (no credentials form), shows graph
metrics from AuraDB, answers a chat query.

### E8. (Optional) Ingest as a Container Apps job

```bash
docker build --platform linux/amd64 -f backend/Dockerfile --target ingest \
  -t <acr>.azurecr.io/graphrag-ingest:latest . && docker push <acr>.azurecr.io/graphrag-ingest:latest

az containerapp job create \
  -n caj-graphrag-ingest -g rg-graphrag --environment cae-graphrag \
  --image <acr>.azurecr.io/graphrag-ingest:latest \
  --registry-server <acr>.azurecr.io --registry-identity "$MI_ID" \
  --user-assigned "$MI_ID" --trigger-type Manual --replica-timeout 3600 \
  --secrets \
    "neo4j-uri=$(kv neo4j-uri)" "neo4j-password=$(kv neo4j-password)" \
    "google-api-key=$(kv google-api-key)" "github-token=$(kv github-token)" \
  --env-vars \
    NEO4J_URI=secretref:neo4j-uri NEO4J_USER=<id> \
    NEO4J_PASSWORD=secretref:neo4j-password \
    GOOGLE_API_KEY=secretref:google-api-key \
    GITHUB_TOKEN=secretref:github-token TARGET_REPO=owner/repo

az containerapp job start -n caj-graphrag-ingest -g rg-graphrag
```

---

## Part F — CI/CD (GitHub Actions)

Tenant blocks service principals, so **Azure OIDC federates to the user-assigned
managed identity** `id-graphrag`.

### F1. Federated credential + roles on the managed identity

```bash
az identity federated-credential create \
  --identity-name id-graphrag --resource-group rg-graphrag --name gh-main \
  --issuer https://token.actions.githubusercontent.com \
  --subject repo:<owner>/<repo>:ref:refs/heads/main \
  --audiences api://AzureADTokenExchange

MI_PRINCIPAL=$(az identity show -n id-graphrag -g rg-graphrag --query principalId -o tsv)
az role assignment create --assignee-object-id "$MI_PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role AcrPush --scope $(az acr show -n <acr> --query id -o tsv)
az role assignment create --assignee-object-id "$MI_PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role Contributor --scope $(az containerapp show -n ca-graphrag-api -g rg-graphrag --query id -o tsv)
```

> Add one federated credential per branch/subject you deploy from. For PR builds:
> `subject repo:<owner>/<repo>:pull_request`.

### F2. GitHub repo → Settings → Secrets and variables → Actions

| Kind | Name | Value |
| --- | --- | --- |
| secret | `AZURE_CLIENT_ID` | managed identity **clientId** (`az identity show … --query clientId`) |
| secret | `AZURE_TENANT_ID` | `az account show --query tenantId -o tsv` |
| secret | `AZURE_SUBSCRIPTION_ID` | `az account show --query id -o tsv` |
| secret | `AZURE_STATIC_WEB_APPS_API_TOKEN` | contents of `.swa-token` |
| secret | `NEO4J_URI` | `neo4j+s://<id>.databases.neo4j.io` (for the keep-alive workflow) |
| secret | `NEO4J_USER` | `<id>` |
| secret | `NEO4J_PASSWORD` | AuraDB password |
| variable | `ACR_NAME` | `<acr>` |
| variable | `ACR_LOGIN_SERVER` | `<acr>.azurecr.io` |
| variable | `CONTAINERAPP_NAME` | `ca-graphrag-api` |
| variable | `RESOURCE_GROUP` | `rg-graphrag` |
| variable | `VITE_API_BASE` | `https://<backend-fqdn>/api` |

### F3. Workflows in the repo

| File | Trigger | Does |
| --- | --- | --- |
| [`backend.yml`](../../.github/workflows/backend.yml) | push to `main` under `backend/**`, `ingest/**`, `requirements*.txt` | OIDC login → `docker build`/`push` `graphrag-api:<sha>` → `az containerapp update` |
| [`azure-static-web-apps.yml`](../../.github/workflows/azure-static-web-apps.yml) | push to `main` under `frontend/**` | `npm ci && npm run build` (with `VITE_API_BASE`) → deploy `frontend/dist` via SWA token |
| [`aura-keepalive.yml`](../../.github/workflows/aura-keepalive.yml) | daily cron | `RETURN 1` against AuraDB so Free tier never auto-pauses |

---

## Operational notes

- **AuraDB Free auto-pauses after ~3 days idle.** [`.github/workflows/aura-keepalive.yml`](../../.github/workflows/aura-keepalive.yml)
  runs a daily no-op query to keep it warm. It needs repo **secrets** `NEO4J_URI`,
  `NEO4J_PASSWORD` (and optionally `NEO4J_USER`). A paused instance refuses connections
  until resumed from the console — that's the pause, not a bug.
- **Costs:** AuraDB Free \$0 · SWA Free \$0 · ACR Basic + Key Vault ≈ a few \$/mo ·
  Container Apps with `--min-replicas 1` ≈ small monthly (set to `0` to scale to zero
  at the cost of cold starts). All charged against the \$100 Azure-for-Students credit.
- **Secrets:** Key Vault (`gr-kv-<suffix>`) is the source of truth for `neo4j-*`,
  `google-api-key`, `github-token`. Because this subscription's Container Apps env
  rejects `keyvaultref:` secrets, `az containerapp create/update` reads the values from
  Key Vault and sets them as **inline** container secrets — so rotating a secret means
  re-running the update (or the `backend.yml` workflow after adding a resync step).
  Nothing secret is in git: confirm `git log -- .env` is empty, and `.azure-deploy.env`
  / `.swa-token` are git-ignored.
- **To redeploy the backend by hand:** `scripts/deploy-backend.sh` (build amd64 → push
  ACR → `az containerapp update`).
