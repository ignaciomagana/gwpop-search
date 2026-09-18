"""Optional agent proposal layer above the deterministic search core."""

from .contracts import (
    AgentDecision,
    AgentRole,
    AgentTaskKind,
    AgentTaskProposal,
    validate_agent_proposal,
)
from .gateway import (
    append_agent_log,
    review_agent_proposal,
    task_id_for_proposal,
)

__all__ = [
    "AgentDecision",
    "AgentRole",
    "AgentTaskKind",
    "AgentTaskProposal",
    "append_agent_log",
    "review_agent_proposal",
    "task_id_for_proposal",
    "validate_agent_proposal",
]
