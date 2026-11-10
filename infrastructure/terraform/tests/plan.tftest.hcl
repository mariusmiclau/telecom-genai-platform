# Mocked providers guarantee these plans cannot create or contact Azure resources.
mock_provider "azurerm" {}
mock_provider "random" {}

variables {
  subscription_id = "00000000-0000-0000-0000-000000000000"
}

run "bootstrap_has_no_workloads" {
  command = plan
  assert {
    condition     = length(azurerm_container_app.api) == 0 && length(azurerm_container_app_job.publisher) == 0
    error_message = "Bootstrap must not start the API or publisher before data is accepted."
  }
  assert {
    condition     = azurerm_postgresql_flexible_server.main.public_network_access_enabled == false
    error_message = "PostgreSQL must remain private."
  }
}

run "workloads_use_secrets_and_private_ingress" {
  command = plan
  variables {
    deploy_api       = true
    deploy_publisher = true
    container_image  = "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    key_vault_id     = "/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/test/providers/Microsoft.KeyVault/vaults/test"
    secret_ids = {
      api-token   = "https://test.vault.azure.net/secrets/api-token"
      admin-token = "https://test.vault.azure.net/secrets/admin-token"
      openai-key  = "https://test.vault.azure.net/secrets/openai-key"
    }
  }
  assert {
    condition     = azurerm_container_app.api[0].ingress[0].external_enabled == false
    error_message = "Ingress must require an explicit external-access choice."
  }
  assert {
    condition     = azurerm_container_app.api[0].template[0].max_replicas == 1
    error_message = "The bounded operator runtime uses one replica."
  }
  assert {
    condition     = azurerm_container_app_job.publisher[0].replica_retry_limit == 0
    error_message = "Publication must not silently replay a completed immutable version."
  }
}
