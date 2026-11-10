locals {
  name             = "${var.project_name}-${random_string.suffix.result}"
  workload_enabled = var.deploy_api || var.deploy_publisher
  secret_names     = { API_TOKEN = "api-token", ADMIN_TOKEN = "admin-token", OPENAI_API_KEY = "openai-key" }
  environment = {
    APP_MODE                           = "live", KNOWLEDGE_VERSION = var.knowledge_version,
    OPENAI_MODEL                       = var.openai_model, EMBEDDING_MODEL_NAME = var.embedding_model,
    EMBEDDING_DIMENSIONS               = tostring(var.embedding_dimensions),
    OTEL_CAPTURE_CONTENT               = "false", OTEL_TRACES_EXPORTER = var.traces_exporter,
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT = var.traces_endpoint,
    DEPLOYMENT_ENVIRONMENT             = "production",
    CONVERSATION_RETENTION_DAYS        = "30"
  }
  database_url = "postgresql://genai:${urlencode(random_password.database.result)}@${azurerm_postgresql_flexible_server.main.fqdn}:5432/telecom_genai?sslmode=verify-full&sslrootcert=/etc/ssl/certs/ca-certificates.crt"
}

resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}
resource "random_password" "database" {
  length  = 40
  special = false
}
resource "azurerm_resource_group" "main" {
  name     = "${local.name}-rg"
  location = var.location
}
resource "azurerm_virtual_network" "main" {
  name                = "${local.name}-vnet"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  address_space       = ["10.41.0.0/16"]
}
resource "azurerm_subnet" "apps" {
  name                 = "apps"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.41.0.0/23"]
  delegation {
    name = "apps"
    service_delegation {
      name    = "Microsoft.App/environments"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
}
resource "azurerm_subnet" "database" {
  name                 = "database"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.41.2.0/24"]
  delegation {
    name = "database"
    service_delegation {
      name    = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
}
resource "azurerm_private_dns_zone" "database" {
  name                = "${local.name}.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name
}
resource "azurerm_private_dns_zone_virtual_network_link" "database" {
  name                  = "database"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.database.name
  virtual_network_id    = azurerm_virtual_network.main.id
}
resource "azurerm_postgresql_flexible_server" "main" {
  name                          = "${local.name}-pg"
  resource_group_name           = azurerm_resource_group.main.name
  location                      = var.location
  version                       = "16"
  administrator_login           = "genai"
  administrator_password        = random_password.database.result
  delegated_subnet_id           = azurerm_subnet.database.id
  private_dns_zone_id           = azurerm_private_dns_zone.database.id
  public_network_access_enabled = false
  sku_name                      = "B_Standard_B1ms"
  storage_mb                    = 32768
  backup_retention_days         = 35
  depends_on                    = [azurerm_private_dns_zone_virtual_network_link.database]
  lifecycle { prevent_destroy = true }
}
resource "azurerm_postgresql_flexible_server_database" "main" {
  name      = "telecom_genai"
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "UTF8"
  collation = "en_US.utf8"
  lifecycle { prevent_destroy = true }
}
resource "azurerm_container_registry" "main" {
  name                = replace(local.name, "-", "")
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  sku                 = "Basic"
  admin_enabled       = false
}
resource "azurerm_user_assigned_identity" "runtime" {
  name                = "${local.name}-runtime"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
}
resource "azurerm_role_assignment" "pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.runtime.principal_id
}
resource "azurerm_role_assignment" "secrets" {
  count                = local.workload_enabled ? 1 : 0
  scope                = var.key_vault_id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.runtime.principal_id
}
resource "azurerm_log_analytics_workspace" "main" {
  name                = "${local.name}-logs"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
}
resource "azurerm_container_app_environment" "main" {
  name                       = "${local.name}-environment"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = var.location
  infrastructure_subnet_id   = azurerm_subnet.apps.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  workload_profile {
    name                  = "Consumption"
    workload_profile_type = "Consumption"
  }
}
resource "azurerm_container_app_job" "publisher" {
  count                        = var.deploy_publisher ? 1 : 0
  name                         = "${local.name}-publish"
  resource_group_name          = azurerm_resource_group.main.name
  location                     = var.location
  container_app_environment_id = azurerm_container_app_environment.main.id
  replica_timeout_in_seconds   = 1800
  replica_retry_limit          = 0
  manual_trigger_config {
    parallelism              = 1
    replica_completion_count = 1
  }
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.runtime.id]
  }
  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.runtime.id
  }
  dynamic "secret" {
    for_each = var.secret_ids
    content {
      name                = secret.key
      key_vault_secret_id = secret.value
      identity            = azurerm_user_assigned_identity.runtime.id
    }
  }
  secret {
    name  = "database-url"
    value = local.database_url
  }
  template {
    container {
      name    = "publisher"
      image   = var.container_image
      cpu     = 0.5
      memory  = "1Gi"
      command = ["python", "-m", "src.operations", "publish"]
      args    = ["--source", var.dataset_path, "--version", var.knowledge_version]
      dynamic "env" {
        for_each = local.environment
        content {
          name  = env.key
          value = env.value
        }
      }
      dynamic "env" {
        for_each = local.secret_names
        content {
          name        = env.key
          secret_name = env.value
        }
      }
      env {
        name        = "DATABASE_URL"
        secret_name = "database-url"
      }
    }
  }
  lifecycle {
    precondition {
      condition     = var.container_image != null && var.key_vault_id != null && alltrue([for name in values(local.secret_names) : contains(keys(var.secret_ids), name)])
      error_message = "Workloads require a pinned image, Key Vault ID and all three secret IDs."
    }
  }
  depends_on = [azurerm_role_assignment.pull, azurerm_role_assignment.secrets, azurerm_postgresql_flexible_server_database.main]
}
resource "azurerm_container_app" "api" {
  count                        = var.deploy_api ? 1 : 0
  name                         = "${local.name}-api"
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.runtime.id]
  }
  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.runtime.id
  }
  dynamic "secret" {
    for_each = var.secret_ids
    content {
      name                = secret.key
      key_vault_secret_id = secret.value
      identity            = azurerm_user_assigned_identity.runtime.id
    }
  }
  secret {
    name  = "database-url"
    value = local.database_url
  }
  ingress {
    external_enabled           = var.external_ingress
    allow_insecure_connections = false
    target_port                = 8000
    transport                  = "http"
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
  template {
    min_replicas = 1
    max_replicas = 1
    container {
      name   = "api"
      image  = var.container_image
      cpu    = 0.5
      memory = "1Gi"
      dynamic "env" {
        for_each = local.environment
        content {
          name  = env.key
          value = env.value
        }
      }
      dynamic "env" {
        for_each = local.secret_names
        content {
          name        = env.key
          secret_name = env.value
        }
      }
      env {
        name        = "DATABASE_URL"
        secret_name = "database-url"
      }
      liveness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/health"
      }
      readiness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/ready"
      }
    }
  }
  lifecycle {
    precondition {
      condition     = var.container_image != null && var.key_vault_id != null && alltrue([for name in values(local.secret_names) : contains(keys(var.secret_ids), name)])
      error_message = "Workloads require a pinned image, Key Vault ID and all three secret IDs. Publish the snapshot before enabling the API."
    }
  }
  depends_on = [azurerm_role_assignment.pull, azurerm_role_assignment.secrets, azurerm_postgresql_flexible_server_database.main]
}
