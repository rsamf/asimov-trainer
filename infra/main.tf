data "azurerm_client_config" "current" {}

locals {
  rg_name = coalesce(var.resource_group_name, "${var.name_prefix}-rg")

  # Storage / Key Vault names must be globally unique and have tight charsets.
  # Suffix with a short hash of subscription+prefix for determinism.
  uniq = substr(sha256("${data.azurerm_client_config.current.subscription_id}-${var.name_prefix}"), 0, 6)
}

resource "azurerm_resource_group" "rg" {
  name     = local.rg_name
  location = var.location
  tags     = var.tags
}

# ---------------------------------------------------------------------------
# Azure Container Registry — holds the training image.
# ---------------------------------------------------------------------------
resource "azurerm_container_registry" "acr" {
  name                = "${var.name_prefix}${local.uniq}"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Basic"
  admin_enabled       = false
  tags                = var.tags
}

# ---------------------------------------------------------------------------
# Azure ML workspace + its mandatory dependencies (storage, KV, AppInsights).
# ---------------------------------------------------------------------------
resource "azurerm_storage_account" "aml" {
  name                     = "${var.name_prefix}sa${local.uniq}"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  tags                     = var.tags
}

resource "azurerm_key_vault" "aml" {
  name                = "${var.name_prefix}-kv-${local.uniq}"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
  purge_protection_enabled = false
  tags                = var.tags
}

resource "azurerm_application_insights" "aml" {
  name                = "${var.name_prefix}-ai-${local.uniq}"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  application_type    = "web"
  tags                = var.tags
}

resource "azurerm_machine_learning_workspace" "aml" {
  name                    = "${var.name_prefix}-aml"
  resource_group_name     = azurerm_resource_group.rg.name
  location                = azurerm_resource_group.rg.location
  application_insights_id = azurerm_application_insights.aml.id
  key_vault_id            = azurerm_key_vault.aml.id
  storage_account_id      = azurerm_storage_account.aml.id
  container_registry_id   = azurerm_container_registry.acr.id

  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

# AML workspace MI pulls images from ACR at job-runtime.
resource "azurerm_role_assignment" "aml_acr_pull" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_machine_learning_workspace.aml.identity[0].principal_id
}

# ---------------------------------------------------------------------------
# GitHub OIDC: an Azure AD app + service principal with federated credential
# bound to a specific GitHub Environment. Branch wildcards (models/*) are
# enforced by GitHub's environment deployment branch rules, not by Azure.
# ---------------------------------------------------------------------------
resource "azuread_application" "github" {
  display_name = "github-${var.name_prefix}"
}

resource "azuread_service_principal" "github" {
  client_id = azuread_application.github.client_id
}

resource "azuread_application_federated_identity_credential" "github_env" {
  application_id = azuread_application.github.id
  display_name   = "github-${var.github_owner}-${var.github_repo}-${var.github_environment}"
  description    = "GitHub Actions OIDC for ${var.github_owner}/${var.github_repo} environment=${var.github_environment}"
  audiences      = ["api://AzureADTokenExchange"]
  issuer         = "https://token.actions.githubusercontent.com"
  subject        = "repo:${var.github_owner}/${var.github_repo}:environment:${var.github_environment}"
}

# CI pushes images to ACR.
resource "azurerm_role_assignment" "github_acr_push" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPush"
  principal_id         = azuread_service_principal.github.object_id
}

# CI submits AML jobs.
resource "azurerm_role_assignment" "github_aml_data_scientist" {
  scope                = azurerm_machine_learning_workspace.aml.id
  role_definition_name = "AzureML Data Scientist"
  principal_id         = azuread_service_principal.github.object_id
}
