# Exported values for Telecom GenAI Platform infrastructure
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------

output "api_endpoint" {
  description = "URL of the deployed API container app"
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}

output "api_internal_endpoint" {
  description = "Internal URL of the API container app"
  value       = "https://${azurerm_container_app.api.name}.internal.${azurerm_container_app_environment.main.default_domain}"
}

# ------------------------------------------------------------------------------
# Azure OpenAI
# ------------------------------------------------------------------------------

output "openai_endpoint" {
  description = "Azure OpenAI service endpoint"
  value       = azurerm_cognitive_account.openai.endpoint
}

output "openai_deployment_name" {
  description = "Name of the GPT model deployment"
  value       = azurerm_cognitive_deployment.gpt4o.name
}

output "openai_embedding_deployment_name" {
  description = "Name of the embedding model deployment"
  value       = azurerm_cognitive_deployment.embedding.name
}

output "openai_resource_name" {
  description = "Name of the Azure OpenAI resource"
  value       = azurerm_cognitive_account.openai.name
}

# ------------------------------------------------------------------------------
# Azure AI Search
# ------------------------------------------------------------------------------

output "search_endpoint" {
  description = "Azure AI Search service endpoint"
  value       = "https://${azurerm_search_service.main.name}.search.windows.net"
}

output "search_service_name" {
  description = "Name of the Azure AI Search service"
  value       = azurerm_search_service.main.name
}

output "search_admin_key" {
  description = "Azure AI Search admin key"
  value       = azurerm_search_service.main.primary_key
  sensitive   = true
}

# ------------------------------------------------------------------------------
# Database
# ------------------------------------------------------------------------------

output "database_host" {
  description = "PostgreSQL server hostname"
  value       = azurerm_postgresql_flexible_server.main.fqdn
}

output "database_name" {
  description = "PostgreSQL database name"
  value       = azurerm_postgresql_flexible_server_database.main.name
}

output "database_connection_string" {
  description = "PostgreSQL connection string (without password)"
  value       = "postgresql://${var.database_admin_username}@${azurerm_postgresql_flexible_server.main.fqdn}:5432/${azurerm_postgresql_flexible_server_database.main.name}?sslmode=require"
  sensitive   = true
}

output "database_connection_string_full" {
  description = "Full PostgreSQL connection string (with password)"
  value       = "postgresql://${var.database_admin_username}:${random_password.database.result}@${azurerm_postgresql_flexible_server.main.fqdn}:5432/${azurerm_postgresql_flexible_server_database.main.name}?sslmode=require"
  sensitive   = true
}

# ------------------------------------------------------------------------------
# Redis
# ------------------------------------------------------------------------------

output "redis_host" {
  description = "Redis cache hostname"
  value       = azurerm_redis_cache.main.hostname
}

output "redis_port" {
  description = "Redis SSL port"
  value       = azurerm_redis_cache.main.ssl_port
}

output "redis_connection_string" {
  description = "Redis connection string"
  value       = "rediss://:${azurerm_redis_cache.main.primary_access_key}@${azurerm_redis_cache.main.hostname}:${azurerm_redis_cache.main.ssl_port}"
  sensitive   = true
}

# ------------------------------------------------------------------------------
# Key Vault
# ------------------------------------------------------------------------------

output "key_vault_uri" {
  description = "Azure Key Vault URI"
  value       = azurerm_key_vault.main.vault_uri
}

output "key_vault_name" {
  description = "Azure Key Vault name"
  value       = azurerm_key_vault.main.name
}

# ------------------------------------------------------------------------------
# Monitoring
# ------------------------------------------------------------------------------

output "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID"
  value       = azurerm_log_analytics_workspace.main.id
}

output "application_insights_connection_string" {
  description = "Application Insights connection string"
  value       = azurerm_application_insights.main.connection_string
  sensitive   = true
}

output "application_insights_instrumentation_key" {
  description = "Application Insights instrumentation key"
  value       = azurerm_application_insights.main.instrumentation_key
  sensitive   = true
}

# ------------------------------------------------------------------------------
# Networking
# ------------------------------------------------------------------------------

output "vnet_id" {
  description = "Virtual Network ID"
  value       = azurerm_virtual_network.main.id
}

output "container_apps_subnet_id" {
  description = "Container Apps subnet ID"
  value       = azurerm_subnet.container_apps.id
}

# ------------------------------------------------------------------------------
# Resource Group
# ------------------------------------------------------------------------------

output "resource_group_name" {
  description = "Resource group name"
  value       = azurerm_resource_group.main.name
}

output "resource_group_location" {
  description = "Resource group location"
  value       = azurerm_resource_group.main.location
}

# ------------------------------------------------------------------------------
# Container App
# ------------------------------------------------------------------------------

output "container_app_identity_principal_id" {
  description = "Container App managed identity principal ID"
  value       = azurerm_container_app.api.identity[0].principal_id
}

output "container_app_environment_id" {
  description = "Container App Environment ID"
  value       = azurerm_container_app_environment.main.id
}

# ------------------------------------------------------------------------------
# Summary Output (for quick reference)
# ------------------------------------------------------------------------------

output "deployment_summary" {
  description = "Summary of deployed resources"
  value = {
    environment  = var.environment
    region       = var.region
    api_url      = "https://${azurerm_container_app.api.ingress[0].fqdn}"
    openai       = azurerm_cognitive_account.openai.endpoint
    search       = "https://${azurerm_search_service.main.name}.search.windows.net"
    database     = azurerm_postgresql_flexible_server.main.fqdn
    redis        = azurerm_redis_cache.main.hostname
    key_vault    = azurerm_key_vault.main.vault_uri
    app_insights = azurerm_application_insights.main.name
  }
}
