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
   Actions → Variables) from `terraform output`:

   - `AZURE_TENANT_ID`
   - `AZURE_SUBSCRIPTION_ID`
   - `AZURE_CLIENT_ID`
   - `AZURE_RESOURCE_GROUP`
   - `AZURE_AML_WORKSPACE`
   - `AZURE_ACR_NAME`

   None are secrets — the OIDC client ID is safe to expose.

6. Push a branch named `models/<something>` and watch
   *Actions → Train Model on Azure ML* fire.

## Local build sanity check

Everything the container needs is committed to this repo — the Asimov URDF /
MJCF / meshes under `data/robot/asimov-v1/`, the retargeted motion `.npz`
under `data/motions/asimov-v1-pyroki-full/`, and the trainer's deps in
`pyproject.toml`. `pyroki` sits in the optional `retarget` dep group and
is intentionally excluded from the image (the trainer reads the
pre-retargeted `.npz`; only the offline retarget CLI needs `pyroki`).

```sh
docker build -t asimov-trainer:test .
docker run --rm --gpus all asimov-trainer:test viewer.kind=null algo.iterations=2
```

To regenerate the retargeted motion `.npz` locally (rare), install the
optional retarget group: `uv sync --group retarget` (resolves the sibling
`../pyroki` editable source declared in `pyproject.toml`).

## Teardown

`terraform destroy`. Note: Azure ML workspaces soft-delete their dependent
Key Vault; recreating with the same name within retention period requires
`az keyvault purge` first.
