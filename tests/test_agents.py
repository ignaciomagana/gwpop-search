import json

import pytest

from gwpop_search.agents import (
    AgentRole,
    AgentTaskKind,
    AgentTaskProposal,
    append_agent_log,
    review_agent_proposal,
    task_id_for_proposal,
    validate_agent_proposal,
)
from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph


def _graph():
    return enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=20,
    )


def test_valid_mutation_proposal_resolves_to_deterministic_task():
    graph = _graph()
    proposal = AgentTaskProposal(
        proposal_id="proposal-1",
        role=AgentRole.SEARCH_PROPOSER,
        kind=AgentTaskKind.PROPOSE_MUTATION,
        payload={
            "parent_hash": graph.root_hash,
            "mutation_id": "chieff.width.linear_q",
            "reason": "residual structure",
        },
        rationale="Review the registered width-q mutation.",
        source_artifacts=("scout-001",),
    )
    validate_agent_proposal(proposal, graph=graph)
    a = review_agent_proposal(proposal, graph=graph, agents_enabled=True)
    b = review_agent_proposal(proposal, graph=graph, agents_enabled=True)

    assert a.accepted
    assert a.resulting_task_id == b.resulting_task_id
    assert a.resulting_task_id == task_id_for_proposal(proposal)


def test_agents_disabled_never_create_task():
    graph = _graph()
    proposal = AgentTaskProposal(
        proposal_id="proposal-2",
        role=AgentRole.NUMERICAL_CRITIC,
        kind=AgentTaskKind.REQUEST_VALIDATION,
        payload={
            "model_hash": graph.root_hash,
            "validation": "importance_ess",
            "reason": "low injection ESS",
        },
        rationale="Request a deterministic ESS validation.",
    )
    decision = review_agent_proposal(
        proposal,
        graph=graph,
        agents_enabled=False,
    )
    assert not decision.accepted
    assert decision.resulting_task_id is None
    assert decision.reason == "agents_disabled"


def test_unknown_model_and_unknown_mutation_are_rejected():
    graph = _graph()
    proposal = AgentTaskProposal(
        proposal_id="bad-mutation",
        role=AgentRole.SEARCH_PROPOSER,
        kind=AgentTaskKind.PROPOSE_MUTATION,
        payload={
            "parent_hash": graph.root_hash,
            "mutation_id": "invent_arbitrary_physics",
            "reason": "no",
        },
        rationale="Attempt an unregistered mutation.",
    )
    decision = review_agent_proposal(proposal, graph=graph, agents_enabled=True)
    assert not decision.accepted
    assert "unregistered mutation" in decision.reason

    unknown = AgentTaskProposal(
        proposal_id="bad-model",
        role=AgentRole.NUMERICAL_CRITIC,
        kind=AgentTaskKind.EVALUATE_MODEL,
        payload={
            "model_hash": "f" * 64,
            "requested_fidelity": "F2",
            "reason": "unknown",
        },
        rationale="Attempt an unknown model.",
    )
    decision2 = review_agent_proposal(unknown, graph=graph, agents_enabled=True)
    assert not decision2.accepted
    assert "unknown model" in decision2.reason


def test_agent_payload_cannot_smuggle_executable_fields():
    with pytest.raises(ValueError, match="unrecognized"):
        AgentTaskProposal(
            proposal_id="exec",
            role=AgentRole.SEARCH_PROPOSER,
            kind=AgentTaskKind.EVALUATE_MODEL,
            payload={
                "model_hash": "a" * 64,
                "requested_fidelity": "F1",
                "reason": "bad",
                "shell_command": "rm -rf /",
            },
            rationale="This should never pass.",
        )


def test_agent_audit_log_is_append_only_jsonl(tmp_path):
    graph = _graph()
    proposal = AgentTaskProposal(
        proposal_id="audit-1",
        role=AgentRole.ADVERSARIAL_VALIDATOR,
        kind=AgentTaskKind.REQUEST_VALIDATION,
        payload={
            "model_hash": graph.root_hash,
            "validation": "null_replay",
            "reason": "calibrate search",
        },
        rationale="Replay the frozen search under its declared null.",
    )
    decision = review_agent_proposal(proposal, graph=graph, agents_enabled=True)
    path = tmp_path / "agents.jsonl"
    append_agent_log(path, proposal, decision)
    append_agent_log(path, proposal, decision)

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == json.loads(lines[1])
    assert json.loads(lines[0])["decision"]["accepted"] is True
