variable "name_prefix" {
  description = "Short prefix used for all resource names (lowercase, no dashes)."
  type        = string
  default     = "asimovtrainer"

  validation {
    condition     = can(regex("^[a-z0-9]{3,18}$", var.name_prefix))
    error_message = "name_prefix must be 3-18 lowercase alphanumeric chars (no dashes)."
  }
}

variable "location" {
  description = "Azure region for all resources."
  type        = string
  default     = "eastus"
}

variable "resource_group_name" {
  description = "Resource group to create. Defaults to <name_prefix>-rg."
  type        = string
  default     = null
}

variable "github_owner" {
  description = "GitHub org or user that owns the asimov-trainer repo."
  type        = string
}

variable "github_repo" {
  description = "GitHub repo name (without owner)."
  type        = string
  default     = "asimov-trainer"
}

variable "github_environment" {
  description = "GitHub Environment name gating the deploy. Must match the workflow's `environment:` field."
  type        = string
  default     = "azure-ml"
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default = {
    project   = "asimov-trainer"
    managedBy = "terraform"
  }
}
