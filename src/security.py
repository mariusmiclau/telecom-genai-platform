"""Verified identities and request-scoped authorization; never trust client filters."""

import asyncio
import secrets
from contextvars import ContextVar
from dataclasses import dataclass

import jwt
from fastapi import HTTPException

from src.settings import AppSettings


@dataclass(frozen=True)
class Principal:
    subject: str = "operator"
    tenant: str = "local"
    roles: frozenset[str] = frozenset({"admin", "reader"})
    groups: frozenset[str] = frozenset()
    enforce_acl: bool = False

    def require(self, permission):
        if permission not in self.roles and "admin" not in self.roles:
            raise HTTPException(403, "Permission denied")


principal_context: ContextVar[Principal] = ContextVar("principal", default=Principal())


def current_principal():
    return principal_context.get()


class Authenticator:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.keys = jwt.PyJWKClient(settings.oidc_jwks_url, timeout=3, lifespan=300) if settings.auth_mode == "oidc" else None

    def decode(self, token):
        if self.keys is None:
            raise jwt.InvalidTokenError("OIDC is not configured")
        header = jwt.get_unverified_header(token)
        if header.get("alg") != "RS256" or not isinstance(header.get("kid"), str) or not 0 < len(header["kid"]) <= 128:
            raise jwt.InvalidTokenError("Unsupported signing algorithm or key identifier")
        key = self.keys.get_signing_key_from_jwt(token)
        claims = jwt.decode(token, key.key, algorithms=["RS256"], audience=self.settings.oidc_audience,
                            issuer=self.settings.oidc_issuer, options={"require": ["exp", "iat", "sub", "iss", "aud"]})
        subject, tenant = claims.get("sub"), claims.get(self.settings.oidc_tenant_claim)
        roles, groups = claims.get("roles", []), claims.get("groups", [])
        if not all(isinstance(value, str) and 0 < len(value) <= 128 for value in (subject, tenant)):
            raise jwt.InvalidTokenError("Missing identity or tenant")
        if tenant == "local":
            raise jwt.InvalidTokenError("The local namespace is reserved for shared-token compatibility")
        if not all(isinstance(values, list) and len(values) <= 128 and all(isinstance(value, str) and 0 < len(value) <= 128 for value in values) for values in (roles, groups)):
            raise jwt.InvalidTokenError("Invalid authorization claims")
        return Principal(subject, tenant, frozenset(roles), frozenset(groups), True)

    async def authenticate(self, token, path):
        if self.settings.auth_mode == "oidc":
            try:
                return await asyncio.to_thread(self.decode, token)
            except (jwt.PyJWTError, ValueError, OSError):
                raise HTTPException(401, "Authentication required") from None
        admin = path.startswith("/api/v1/ingest") or path.startswith("/api/v1/actions") or (path.startswith("/api/v1/feedback") and path.endswith(("/review", "/regression")))
        expected = self.settings.admin_token if admin else self.settings.api_token
        if not expected.get_secret_value() or not secrets.compare_digest(token.encode(), expected.get_secret_value().encode()):
            raise HTTPException(401, "Authentication required")
        return Principal(roles=frozenset({"reader", "admin"} if admin else {"reader"}))


def route_permission(path: str, method: str):
    if path.startswith("/api/v1/feedback") and path.endswith(("/review", "/regression")):
        return "feedback_review"
    if path.startswith("/api/v1/ingest"):
        return "ingest"
    if path.startswith("/api/v1/actions"):
        if path.endswith(("/approve", "/reject")):
            return "action_approve"
        if path.endswith("/execute"):
            return "action_execute"
        if method == "POST":
            return "action_request"
    return "reader"
