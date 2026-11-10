output "resource_group" { value = azurerm_resource_group.main.name }
output "registry" { value = azurerm_container_registry.main.login_server }
output "publisher_job" { value = try(azurerm_container_app_job.publisher[0].name, null) }
output "api_url" { value = try("https://${azurerm_container_app.api[0].latest_revision_fqdn}", null) }
output "database_host" { value = azurerm_postgresql_flexible_server.main.fqdn }
