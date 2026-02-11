# Input variables for Telecom GenAI Platform infrastructure

# ------------------------------------------------------------------------------
# Environment Configuration
# ------------------------------------------------------------------------------

variable "environment" {
  type        = string
  description = "Deployment environment (dev, staging, prod)"
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "region" {
  type        = string
  description = "Azure region for resource deployment"
  default     = "eastus"

  validation {
    condition = contains([
      "eastus", "eastus2", "westus", "westus2", "westus3",
      "centralus", "northcentralus", "southcentralus",
      "westeurope", "northeurope", "uksouth", "ukwest",
      "australiaeast", "southeastasia", "japaneast"
    ], var.region)
    error_message = "Region must be a valid Azure region with OpenAI availability."
  }
}

variable "project_name" {
  type        = string
  description = "Project name used for resource naming"
  default     = "telecom-genai"
}

variable "tags" {
  type        = map(string)
  description = "Tags to apply to all resources"
  default = {
    project     = "telecom-genai-platform"
    managed_by  = "terraform"
    cost_center = "ai-platform"
  }
}

# ------------------------------------------------------------------------------
# Azure OpenAI Configuration
# ------------------------------------------------------------------------------

variable "openai_model_deployment_name" {
  type        = string
  description = "Name for the OpenAI model deployment"
  default     = "gpt-4o"
}

variable "openai_model_name" {
  type        = string
  description = "OpenAI model to deploy"
  default     = "gpt-4o"
}

variable "openai_model_version" {
  type        = string
  description = "Version of the OpenAI model"
  default     = "2024-08-06"
}

variable "openai_embedding_model" {
  type        = string
  description = "Embedding model name"
  default     = "text-embedding-ada-002"
}

variable "openai_embedding_deployment_name" {
  type        = string
  description = "Name for the embedding model deployment"
  default     = "text-embedding-ada-002"
}

variable "openai_capacity" {
  type        = number
  description = "Tokens per minute capacity (in thousands)"
  default     = 30

  validation {
    condition     = var.openai_capacity >= 1 && var.openai_capacity <= 300
    error_message = "OpenAI capacity must be between 1 and 300 (thousands of TPM)."
  }
}

# ------------------------------------------------------------------------------
# Database Configuration
# ------------------------------------------------------------------------------

variable "database_sku" {
  type        = string
  description = "SKU tier for PostgreSQL Flexible Server"
  default     = "B_Standard_B1ms"

  validation {
    condition = contains([
      "B_Standard_B1ms", "B_Standard_B2s",           # Burstable
      "GP_Standard_D2s_v3", "GP_Standard_D4s_v3",   # General Purpose
      "MO_Standard_E2s_v3", "MO_Standard_E4s_v3"    # Memory Optimized
    ], var.database_sku)
    error_message = "Database SKU must be a valid PostgreSQL Flexible Server SKU."
  }
}

variable "database_storage_mb" {
  type        = number
  description = "Database storage size in MB"
  default     = 32768 # 32 GB

  validation {
    condition     = var.database_storage_mb >= 32768 && var.database_storage_mb <= 16777216
    error_message = "Database storage must be between 32 GB and 16 TB."
  }
}

variable "database_version" {
  type        = string
  description = "PostgreSQL version"
  default     = "15"
}

variable "database_admin_username" {
  type        = string
  description = "Database administrator username"
  default     = "pgadmin"
  sensitive   = true
}

# ------------------------------------------------------------------------------
# Redis Configuration
# ------------------------------------------------------------------------------

variable "redis_sku" {
  type        = string
  description = "SKU for Azure Cache for Redis"
  default     = "Basic"

  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.redis_sku)
    error_message = "Redis SKU must be Basic, Standard, or Premium."
  }
}

variable "redis_capacity" {
  type        = number
  description = "Redis cache capacity (0-6 for Basic/Standard, 1-5 for Premium)"
  default     = 0
}

variable "redis_family" {
  type        = string
  description = "Redis SKU family (C for Basic/Standard, P for Premium)"
  default     = "C"
}

# ------------------------------------------------------------------------------
# AI Search Configuration
# ------------------------------------------------------------------------------

variable "search_sku" {
  type        = string
  description = "SKU for Azure AI Search"
  default     = "basic"

  validation {
    condition     = contains(["free", "basic", "standard", "standard2", "standard3"], var.search_sku)
    error_message = "Search SKU must be free, basic, standard, standard2, or standard3."
  }
}

variable "search_replica_count" {
  type        = number
  description = "Number of search replicas"
  default     = 1
}

variable "search_partition_count" {
  type        = number
  description = "Number of search partitions"
  default     = 1
}

# ------------------------------------------------------------------------------
# Container Apps Configuration
# ------------------------------------------------------------------------------

variable "container_app_cpu" {
  type        = number
  description = "CPU allocation for container app (cores)"
  default     = 0.5
}

variable "container_app_memory" {
  type        = string
  description = "Memory allocation for container app"
  default     = "1Gi"
}

variable "container_app_min_replicas" {
  type        = number
  description = "Minimum number of container replicas"
  default     = 1
}

variable "container_app_max_replicas" {
  type        = number
  description = "Maximum number of container replicas"
  default     = 10
}

variable "container_image" {
  type        = string
  description = "Container image for the API"
  default     = "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"
}

# ------------------------------------------------------------------------------
# Networking Configuration
# ------------------------------------------------------------------------------

variable "vnet_address_space" {
  type        = list(string)
  description = "Address space for the virtual network"
  default     = ["10.0.0.0/16"]
}

variable "enable_private_endpoints" {
  type        = bool
  description = "Enable private endpoints for secure connectivity"
  default     = true
}

# ------------------------------------------------------------------------------
# Monitoring Configuration
# ------------------------------------------------------------------------------

variable "log_retention_days" {
  type        = number
  description = "Log Analytics workspace retention in days"
  default     = 30
}

variable "enable_diagnostic_settings" {
  type        = bool
  description = "Enable diagnostic settings for all resources"
  default     = true
}
