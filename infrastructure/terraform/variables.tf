variable "subscription_id" { type = string }
variable "location" {
  type    = string
  default = "uksouth"
}
variable "project_name" {
  type    = string
  default = "telecom-genai"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,16}$", var.project_name))
    error_message = "Use 3-17 lowercase letters, numbers or hyphens, starting with a letter."
  }
}
variable "container_image" {
  type     = string
  default  = null
  nullable = true
  validation {
    condition     = var.container_image == null || can(regex("@sha256:[0-9a-f]{64}$", var.container_image))
    error_message = "An immutable registry image digest is required."
  }
}
variable "deploy_publisher" {
  type    = bool
  default = false
}
variable "deploy_api" {
  type    = bool
  default = false
}
variable "external_ingress" {
  type    = bool
  default = false
}
variable "knowledge_version" {
  type    = string
  default = "tickets-v1"
}
variable "dataset_path" {
  type    = string
  default = "/app/datasets/tickets-v2.0-ticket-aware-chunking/tickets.jsonl"
}
variable "openai_model" {
  type    = string
  default = "gpt-4o"
}
variable "embedding_model" {
  type    = string
  default = "text-embedding-3-small"
}
variable "embedding_dimensions" {
  type    = number
  default = 1536
}
variable "key_vault_id" {
  type    = string
  default = null
}
variable "secret_ids" {
  type        = map(string)
  default     = {}
  description = "Existing Key Vault secret URLs keyed by api-token, admin-token, openai-key."
}
variable "traces_exporter" {
  type    = string
  default = "none"
  validation {
    condition     = contains(["none", "otlp"], var.traces_exporter)
    error_message = "Select none or otlp."
  }
}
variable "traces_endpoint" {
  type        = string
  default     = "http://localhost:4318/v1/traces"
  description = "Approved private OTLP/HTTP collector; used only when traces_exporter is otlp."
}
