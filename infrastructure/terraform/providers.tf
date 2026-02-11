# Azure provider configuration for Telecom GenAI Platform

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.85.0"
    }
    azapi = {
      source  = "azure/azapi"
      version = "~> 1.10.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6.0"
    }
  }

  # Backend configuration for remote state storage
  # Uncomment and configure for production use
  # backend "azurerm" {
  #   resource_group_name  = "tfstate-rg"
  #   storage_account_name = "tfstatetelecomgenai"
  #   container_name       = "tfstate"
  #   key                  = "telecom-genai.tfstate"
  # }
}

# Azure Provider with required features
provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }

    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }

    cognitive_account {
      purge_soft_delete_on_destroy = true
    }

    log_analytics_workspace {
      permanently_delete_on_destroy = true
    }
  }

  # Optional: Skip provider registration if already registered
  skip_provider_registration = true
}

# Azure API provider for preview features
provider "azapi" {}

# Random provider for generating unique names
provider "random" {}
