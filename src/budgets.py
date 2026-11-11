"""Per-request conservative token accounting with no nested hidden retries."""

from contextvars import ContextVar
from dataclasses import asdict, dataclass


class BudgetExceeded(ValueError):  # noqa: N818 - public budget denial signal
    pass


@dataclass
class Consumption:
    input_token_upper_bound: int = 0
    output_token_limit: int = 0
    provider_reported_tokens: int = 0
    embedding_token_upper_bound: int = 0
    completion_attempts: int = 0
    reserved_token_upper_bound: int = 0
    maximum_request_tokens: int = 24000
    prompt_token_limit: int = 16000
    history_token_limit: int = 2000
    retry_limit: int = 0

    def reserve(self, tokens):
        if self.reserved_token_upper_bound + tokens > self.maximum_request_tokens:
            raise BudgetExceeded("Per-request token budget exceeded")
        self.reserved_token_upper_bound += tokens

    def report(self):
        return {**asdict(self), "accounting": "utf8_bytes_plus_message_overhead_upper_bound",
                "currency_cost": None, "currency_cost_reason": "No billing tariff configured; reservations are not actual billing."}


consumption_context: ContextVar[Consumption | None] = ContextVar("consumption", default=None)
prompt_record_context: ContextVar[dict | None] = ContextVar("prompt_record", default=None)


def token_bound(text):
    # Conservative for byte-level tokenizers, including non-ASCII inputs.
    return len(text.encode("utf-8")) + 16


def messages_bound(messages):
    return 16 + sum(token_bound(item["content"]) + token_bound(item["role"]) for item in messages)
