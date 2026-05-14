# Values to paste into the GitHub repo's Actions Variables after `terraform apply`.
# Settings → Secrets and variables → Actions → Variables.

output "AZURE_TENANT_ID" {
  value = data.azurerm_client_config.current.tenant_id
}

output "AZURE_SUBSCRIPTION_ID" {
  value = data.azurerm_client_config.current.subscription_id
}

output "AZURE_CLIENT_ID" {
  description = "GitHub-OIDC service principal client ID."
  value       = azuread_application.github.client_id
}

output "AZURE_RESOURCE_GROUP" {
  value = azurerm_resource_group.rg.name
}

output "AZURE_AML_WORKSPACE" {
  value = azurerm_machine_learning_workspace.aml.name
}

output "AZURE_ACR_NAME" {
  value = azurerm_container_registry.acr.name
}

output "azure_environment_name" {
  description = "Must match the GitHub Environment name and the workflow's environment field."
  value       = var.github_environment
}

output "acr_login_server" {
  value = azurerm_container_registry.acr.login_server
}
