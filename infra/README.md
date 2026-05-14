# `infra/` — Terraform stack for Azure ML training

Stands up everything the `train_model.yml` workflow needs:

- Resource group, Azure Container Registry (Basic).
- Azure ML workspace + its mandatory deps (storage account, Key Vault, App Insights).
- `AcrPull` for the AML workspace's managed identity (job-time image pull).
- A GitHub-OIDC Azure AD application + service principal with a federated
  credential bound to a GitHub Environment, granted `AcrPush` on the ACR and
  `AzureML Data Scientist` on the workspace.

## One-time bootstrap

1. `az login` (interactive) — the operator running `terraform apply` needs
   `Owner` or `User Access Administrator` on the target subscription so the
   role assignments succeed.

2. Create `terraform.tfvars`:

   ```hcl
   name_prefix  = "asimovtrainer"   # 3-18 lowercase alphanumerics
   location     = "eastus"
   github_owner = "<your-gh-org-or-user>"
   github_repo  = "asimov-trainer"
   # github_environment defaults to "azure-ml" — change it if you want a different
   # GitHub Environment name. Whatever you set MUST match the `environment:` field
   # in .github/workflows/train_model.yml.
   ```

3. `terraform init && terraform apply`. Confirm.

4. **Create the GitHub Environment** in repo settings → Environments → New
   environment → name `azure-ml` (or whatever you set above). Under
   *Deployment branches and tags*, add rules for `models` and `models/*`.
   Terraform can't manage GitHub itself without an additional provider, so
   this is the one manual step.

5. Set GitHub repo Actions *Variables* (Settings → Secrets and variables →
   Actions → Variables):

   From `terraform output`:
   - `AZURE_TENANT_ID`
   - `AZURE_SUBSCRIPTION_ID`
   - `AZURE_CLIENT_ID`
   - `AZURE_RESOURCE_GROUP`
   - `AZURE_AML_WORKSPACE`
   - `AZURE_ACR_NAME`

   Set manually — the sibling repos the build needs to clone:
   - `PYROKI_REPO`            — e.g. `your-org/pyroki`
   - `LOCAL_ROBOT_FILES_REPO` — e.g. `your-org/local-robot-files`

   Optionally, if either sibling repo is private and the default
   `GITHUB_TOKEN` can't read it (different org), add a *Secret* named
   `GH_VENDOR_PAT` — a PAT or fine-grained token with read access on those
   repos. The workflow falls back to `github.token` when the secret is unset.

   None of the variables are secrets — the OIDC client ID is safe to expose.

6. Push a branch named `models/<something>` and watch
   *Actions → Train Model on Azure ML* fire.

## Build-context prerequisites for the Docker image

The Dockerfile expects two paths inside the build context that don't live
naturally in this repo:

- `./pyroki/` — sibling editable repo. The CI workflow clones it via a
  second `actions/checkout` step (see `vars.PYROKI_REPO`). Locally, the
  `deploy/prepare-build-context.sh` script copies it from `../pyroki/`.

- `./data/robot/asimov-v1/` — currently a symlink into `local-robot-files/`.
  The prep script materialises a real copy. CI clones the sibling repo
  (`vars.LOCAL_ROBOT_FILES_REPO`) and points `LOCAL_ROBOT_FILES_SRC` at it.

To build locally:

```sh
deploy/prepare-build-context.sh
docker build -t asimov-trainer:test .
docker run --rm --gpus all asimov-trainer:test viewer.kind=null algo.iterations=2
```

The prep script is idempotent — re-running it is a no-op if the build
context is already materialised.

## Teardown

`terraform destroy`. Note: Azure ML workspaces soft-delete their dependent
Key Vault; recreating with the same name within retention period requires
`az keyvault purge` first.
