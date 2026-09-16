from copy import deepcopy
from types import SimpleNamespace
from typing import Callable

import pytest

from tau2.agent.base_agent import AgentError
from tau2.agent.llm_agent import LLMAgent, LLMSoloAgent
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.simulation import TerminationReason
from tau2.data_model.tasks import EnvAssertion, InitialState, Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.orchestrator import (
    DEFAULT_FIRST_AGENT_MESSAGE,
    Orchestrator,
    Role,
    get_finish_reason,
)
from tau2.user.user_simulator import DummyUser, UserSimulator
from tau2.utils.llm_utils import ToolCallArgumentsError


@pytest.fixture
def user_simulator() -> UserSimulator:
    return UserSimulator(
        instructions="You are a user simulator.",
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
    )


@pytest.fixture
def dummy_user() -> DummyUser:
    return DummyUser()


@pytest.fixture
def agent(get_environment: Callable[[], Environment]) -> LLMAgent:
    environment = get_environment()
    return LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
    )


@pytest.fixture
def solo_agent(
    get_environment: Callable[[], Environment], base_task: Task
) -> LLMSoloAgent:
    environment = get_environment()
    return LLMSoloAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        task=base_task,
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
    )


def test_orchestrator_initialize_base(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    get_environment: Callable[[], Environment],
    base_task: Task,
):
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    # Check Initialization
    assert orchestrator.from_role == Role.AGENT
    assert orchestrator.to_role == Role.USER
    assert orchestrator.step_count == 0
    assert not orchestrator.done
    assert orchestrator.termination_reason is None
    assert len(orchestrator.trajectory) == 1
    assert isinstance(orchestrator.trajectory[0], AssistantMessage)
    assert orchestrator.trajectory[0].content == DEFAULT_FIRST_AGENT_MESSAGE.content
    assert orchestrator.message.content == DEFAULT_FIRST_AGENT_MESSAGE.content


def test_orchestrator_initialize_with_message_history(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    get_environment: Callable[[], Environment],
    task_with_message_history: Task,
):
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=task_with_message_history,
    )
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_number_of_tasks",
            arguments={"user_id": "user_1", "expected_number": 1},
        )
    )
    orchestrator.initialize()
    assert orchestrator.from_role == Role.AGENT
    assert orchestrator.to_role == Role.USER
    assert orchestrator.step_count == 0
    assert not orchestrator.done
    assert orchestrator.termination_reason is None
    assert len(orchestrator.get_trajectory()) == len(
        task_with_message_history.initial_state.message_history
    )

    user_state = orchestrator.user_state
    print(user_state.model_dump_json(indent=2))
    assert len(user_state.messages) == 1

    agent_state = orchestrator.agent_state
    print(agent_state.model_dump_json(indent=2))
    assert len(agent_state.messages) == len(
        task_with_message_history.initial_state.message_history
    )
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_task_status",
            arguments={"task_id": "task_2", "expected_status": "pending"},
        )
    )
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_number_of_tasks",
            arguments={"user_id": "user_1", "expected_number": 2},
        )
    )


def test_orchestrator_initialize_with_initialization_data(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    get_environment: Callable[[], Environment],
    task_with_initialization_data: Task,
):
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=task_with_initialization_data,
    )
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_number_of_tasks",
            arguments={"user_id": "user_1", "expected_number": 1},
        )
    )
    orchestrator.initialize()
    print(orchestrator.environment.tools.db.model_dump_json(indent=2))
    assert orchestrator.from_role == Role.AGENT
    assert orchestrator.to_role == Role.USER
    assert orchestrator.step_count == 0
    assert not orchestrator.done
    assert orchestrator.termination_reason is None
    assert len(orchestrator.get_trajectory()) == 1
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_task_status",
            arguments={"task_id": "task_2", "expected_status": "pending"},
        )
    )
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_number_of_tasks",
            arguments={"user_id": "user_1", "expected_number": 2},
        )
    )


def test_orchestrator_initialize_with_initialization_actions(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    get_environment: Callable[[], Environment],
    task_with_initialization_actions: Task,
):
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=task_with_initialization_actions,
    )
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_number_of_tasks",
            arguments={"user_id": "user_1", "expected_number": 1},
        )
    )
    orchestrator.initialize()
    print(orchestrator.environment.tools.db.model_dump_json(indent=2))
    assert orchestrator.from_role == Role.AGENT
    assert orchestrator.to_role == Role.USER
    assert orchestrator.step_count == 0
    assert not orchestrator.done
    assert orchestrator.termination_reason is None
    assert len(orchestrator.get_trajectory()) == 1
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_task_status",
            arguments={"task_id": "task_2", "expected_status": "pending"},
        )
    )
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_number_of_tasks",
            arguments={"user_id": "user_1", "expected_number": 2},
        )
    )


def test_orchestrator_step(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    # Check Step 1
    orchestrator.step()
    assert orchestrator.from_role == Role.USER
    assert orchestrator.to_role == Role.AGENT
    assert orchestrator.step_count == 1
    assert not orchestrator.done
    assert orchestrator.termination_reason is None
    assert len(orchestrator.get_trajectory()) == 2
    assert isinstance(orchestrator.get_trajectory()[1], UserMessage)
    assert isinstance(orchestrator.message, UserMessage)

    # Check Step 2
    orchestrator.step()
    assert orchestrator.from_role == Role.AGENT
    assert orchestrator.to_role in [Role.ENV, Role.USER]
    assert orchestrator.step_count == 2
    assert not orchestrator.done
    assert orchestrator.termination_reason is None
    assert len(orchestrator.get_trajectory()) == 3
    assert isinstance(orchestrator.get_trajectory()[2], AssistantMessage)
    assert isinstance(orchestrator.message, AssistantMessage)


def test_orchestrator_restart(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    orchestrator1 = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
        seed=300,
    )
    orchestrator1.initialize()
    # Create a partial message history
    for _ in range(3):
        orchestrator1.step()
    partial_message_history = orchestrator1.get_trajectory()

    # Create a new task with the partial message history
    task2 = deepcopy(base_task)
    initial_state = InitialState(
        message_history=partial_message_history,
        variables={},
        state={},
    )
    task2.initial_state = initial_state
    # Create a new orchestrator with the partial new task
    orchestrator2 = Orchestrator(
        domain=domain_name,
        environment=get_environment(),
        user=user_simulator,
        agent=agent,
        task=task2,
        seed=300,
    )
    orchestrator2.initialize()

    assert orchestrator1.to_role == orchestrator2.to_role
    assert orchestrator1.from_role == orchestrator2.from_role
    assert orchestrator1.message.content == orchestrator2.message.content
    for msg1, msg2 in zip(
        orchestrator1.get_trajectory(), orchestrator2.get_trajectory()
    ):
        assert msg1.content == msg2.content

    ## Step each orchestrator 3 times
    for _ in range(3):
        if not orchestrator1.done:
            orchestrator1.step()
        if not orchestrator2.done:
            orchestrator2.step()
        print("--------------------------------")
        print("Orchestrator 1")
        print(orchestrator1.message)
        print("--------------------------------")
        print("Orchestrator 2")
        print(orchestrator2.message)
        print("--------------------------------")


def test_orchestrator_run(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    orchestrator = Orchestrator(
        domain=domain_name,
        environment=get_environment(),
        user=user_simulator,
        agent=agent,
        task=base_task,
        max_steps=10,
    )
    simulation_run = orchestrator.run()
    assert simulation_run is not None


def test_orchestrator_run_with_solo_agent(
    domain_name: str,
    dummy_user: DummyUser,
    solo_agent: LLMSoloAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    orchestrator = Orchestrator(
        domain=domain_name,
        environment=get_environment(solo_mode=True),
        user=dummy_user,
        agent=solo_agent,
        task=base_task,
        max_steps=10,
        solo_mode=True,
    )
    simulation_run = orchestrator.run()
    assert simulation_run is not None

    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_task_status",
            arguments={"task_id": "task_2", "expected_status": "pending"},
        )
    )


def test_validate_communication_default_is_false(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that validate_communication defaults to False for backwards compatibility."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    assert orchestrator.validate_communication is False


def test_validate_communication_enabled(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that validate_communication can be enabled."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
        validate_communication=True,
    )
    assert orchestrator.validate_communication is True


def test_validate_communication_catches_empty_message(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that empty messages are caught when validation is enabled."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
        validate_communication=True,
    )
    orchestrator.initialize()

    # Manually set up orchestrator state with empty message
    orchestrator.from_role = Role.AGENT
    orchestrator.message = AssistantMessage(role="assistant", content="", cost=0.0)
    orchestrator.done = False

    # Check communication should catch the empty message
    orchestrator.check_communication_error()

    # Should terminate due to agent error (empty message)
    assert orchestrator.done is True
    assert orchestrator.termination_reason == TerminationReason.AGENT_ERROR


def test_validate_communication_catches_mixed_message(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that mixed messages (text + tool calls) are caught when validation is enabled."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
        validate_communication=True,
    )
    orchestrator.initialize()

    # Manually set up orchestrator state with mixed message (text + tool call)
    orchestrator.from_role = Role.AGENT
    orchestrator.message = AssistantMessage(
        role="assistant",
        content="I'll help you with that",
        tool_calls=[ToolCall(id="1", name="search", arguments={})],
        cost=0.0,
    )
    orchestrator.done = False

    # Check communication should catch the mixed message
    orchestrator.check_communication_error()

    # Should terminate due to agent error (mixed message)
    assert orchestrator.done is True
    assert orchestrator.termination_reason == TerminationReason.AGENT_ERROR


def test_agent_stop_on_last_step_is_not_relabelled_max_steps(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that a stop on the last allowed step keeps AGENT_STOP instead of MAX_STEPS."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
        max_steps=4,
    )
    orchestrator.initialize()

    stop_message = AssistantMessage(role="assistant", content="###STOP###", cost=0.0)
    orchestrator.agent.generate_next_message = lambda message, state: (
        stop_message,
        state,
    )
    orchestrator.agent.is_stop = lambda message: message is stop_message
    # Hand the turn to the agent on the last step run() would allow.
    orchestrator.from_role = Role.USER
    orchestrator.to_role = Role.AGENT
    orchestrator.step_count = orchestrator.max_steps - 1

    # run() calls _check_termination() after every step, including the one that sets done.
    orchestrator.step()
    orchestrator._check_termination()

    assert orchestrator.step_count == orchestrator.max_steps
    assert orchestrator.done is True
    assert orchestrator.termination_reason == TerminationReason.AGENT_STOP


def test_full_duplex_termination_reason_is_not_relabelled_max_steps(
    domain_name: str,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that a set termination reason survives the full-duplex termination check."""
    # Real streaming participants need audio dependencies this suite cannot import, and the
    # check under test reads only the counters.
    streaming_stub = SimpleNamespace(get_next_chunk=lambda *args, **kwargs: None)
    orchestrator = FullDuplexOrchestrator(
        domain=domain_name,
        user=streaming_stub,
        agent=streaming_stub,
        environment=get_environment(),
        task=base_task,
        max_steps=4,
    )
    orchestrator.step_count = orchestrator.max_steps
    orchestrator.done = True
    orchestrator.termination_reason = TerminationReason.AGENT_STOP

    orchestrator._check_termination()

    assert orchestrator.termination_reason == TerminationReason.AGENT_STOP


def test_empty_agent_message_terminates_with_agent_error(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that an empty agent message ends the simulation instead of raising."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    empty_message = AssistantMessage(role="assistant", content=None, cost=0.0)
    orchestrator.agent.generate_next_message = lambda message, state: (
        empty_message,
        state,
    )
    # Hand the turn to the agent.
    orchestrator.from_role = Role.USER
    orchestrator.to_role = Role.AGENT

    orchestrator.step()  # Must not raise.

    assert orchestrator.done is True
    assert orchestrator.termination_reason == TerminationReason.AGENT_ERROR
    assert orchestrator.from_role == Role.AGENT
    assert orchestrator.to_role == Role.USER
    assert orchestrator.message is empty_message
    assert orchestrator.get_trajectory()[-1].content is None


def test_empty_agent_message_on_last_step_keeps_agent_error(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that an empty agent message on the last allowed step is not relabelled MAX_STEPS."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
        max_steps=4,
    )
    orchestrator.initialize()

    empty_message = AssistantMessage(role="assistant", content=None, cost=0.0)
    orchestrator.agent.generate_next_message = lambda message, state: (
        empty_message,
        state,
    )
    # Hand the turn to the agent on the last step run() would allow.
    orchestrator.from_role = Role.USER
    orchestrator.to_role = Role.AGENT
    orchestrator.step_count = orchestrator.max_steps - 1

    # run() calls _check_termination() after every step, including the one that sets done.
    orchestrator.step()
    orchestrator._check_termination()

    assert orchestrator.step_count == orchestrator.max_steps
    assert orchestrator.done is True
    assert orchestrator.termination_reason == TerminationReason.AGENT_ERROR


def test_agent_message_with_empty_tool_calls_terminates_with_agent_error(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that an empty tool call list is an agent error rather than a call to the environment."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    # A "tool_calls" finish_reason must still be charged: it is the label our endpoints put on
    # most ordinary turns, so retrying on it would restore the re-roll bias.
    empty_message = AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[],
        cost=0.0,
        raw_data={"choices": [{"finish_reason": "tool_calls"}]},
    )
    orchestrator.agent.generate_next_message = lambda message, state: (
        empty_message,
        state,
    )
    orchestrator.from_role = Role.USER
    orchestrator.to_role = Role.AGENT

    orchestrator.step()

    assert orchestrator.done is True
    assert orchestrator.termination_reason == TerminationReason.AGENT_ERROR
    assert orchestrator.to_role == Role.USER


def test_get_finish_reason_tolerates_an_unreadable_provider_response():
    """Test that an unrecognised raw_data shape yields None instead of raising."""

    def message(raw_data):
        return AssistantMessage(
            role="assistant", content="Hi", cost=0.0, raw_data=raw_data
        )

    assert (
        get_finish_reason(message({"choices": [{"finish_reason": "length"}]}))
        == "length"
    )
    # This runs inside step()'s exception handler, where a raise would turn a charged turn
    # back into a retry.
    assert get_finish_reason(None) is None
    assert get_finish_reason(message(None)) is None
    assert get_finish_reason(message({"choices": []})) is None
    assert get_finish_reason(message({"choices": ["stop"]})) is None
    assert get_finish_reason(message({"choices": {"a": 1}})) is None


def test_content_filter_empty_agent_message_is_left_to_the_caller(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that a provider content filter trip is raised so the caller can retry it."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    filtered_message = AssistantMessage(
        role="assistant",
        content=None,
        cost=0.0,
        raw_data={"choices": [{"finish_reason": "content_filter"}]},
    )
    orchestrator.agent.generate_next_message = lambda message, state: (
        filtered_message,
        state,
    )
    orchestrator.from_role = Role.USER
    orchestrator.to_role = Role.AGENT

    with pytest.raises(AgentError):
        orchestrator.step()

    assert orchestrator.done is False
    assert orchestrator.termination_reason is None


def test_agent_error_without_a_message_terminates_with_agent_error(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that an AgentError raised before a message exists ends the simulation."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    def raise_agent_error(message, state):
        raise AgentError("Model returned invalid JSON arguments for tool get_task")

    orchestrator.agent.generate_next_message = raise_agent_error
    orchestrator.from_role = Role.USER
    orchestrator.to_role = Role.AGENT
    trajectory_length = len(orchestrator.trajectory)

    orchestrator.step()  # Must not raise UnboundLocalError.

    assert orchestrator.done is True
    assert orchestrator.termination_reason == TerminationReason.AGENT_ERROR
    assert len(orchestrator.trajectory) == trajectory_length
    # No agent message exists, so the turn still belongs to the agent and _finalize() hands
    # the user's message to agent.stop().
    assert orchestrator.to_role == Role.AGENT


def test_agent_error_is_not_relabelled_user_error_when_validating(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that a failed agent turn is not billed to the user simulator."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
        validate_communication=True,
    )
    orchestrator.initialize()

    def raise_agent_error(message, state):
        raise AgentError("Model returned invalid JSON arguments for tool get_task")

    orchestrator.agent.generate_next_message = raise_agent_error
    # With no agent message the pending message stays the user's, and this one breaks the
    # protocol, so re-running the check at the end of step() would charge the user instead.
    orchestrator.message = UserMessage(
        role="user",
        content="Book it.",
        tool_calls=[ToolCall(id="1", name="get_task", arguments={"task_id": "task_1"})],
        cost=0.0,
    )
    orchestrator.from_role = Role.USER
    orchestrator.to_role = Role.AGENT

    orchestrator.step()

    assert orchestrator.done is True
    assert orchestrator.termination_reason == TerminationReason.AGENT_ERROR


def test_tool_call_arguments_error_is_reported_as_agent_error(agent: LLMAgent):
    """Test that malformed tool call arguments reach the orchestrator as an agent failure."""

    def raise_parse_error(message, state):
        raise ToolCallArgumentsError(
            "Model returned invalid JSON arguments for tool get_task"
        )

    agent._generate_next_message = raise_parse_error
    state = agent.get_init_state()

    with pytest.raises(AgentError):
        agent.generate_next_message(UserMessage(role="user", content="Hi"), state)


def test_solo_mode_message_to_user_terminates_with_agent_error(
    domain_name: str,
    dummy_user: DummyUser,
    solo_agent: LLMSoloAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that a solo-mode agent message addressed to the user is an agent error."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=dummy_user,
        agent=solo_agent,
        environment=get_environment(solo_mode=True),
        task=base_task,
        solo_mode=True,
    )
    tool_call_message = AssistantMessage(
        role="assistant",
        tool_calls=[ToolCall(id="1", name="get_task", arguments={"task_id": "task_1"})],
        cost=0.0,
    )
    orchestrator.agent.generate_next_message = lambda message, state: (
        tool_call_message,
        state,
    )
    orchestrator.initialize()
    assert orchestrator.to_role == Role.ENV

    # The agent is mocked here, so this pins the orchestrator's solo branch; that LLMSoloAgent
    # returns such a message instead of raising is covered in tests/test_agent.py.
    text_message = AssistantMessage(
        role="assistant", content="Could you confirm?", cost=0.0
    )
    orchestrator.agent.generate_next_message = lambda message, state: (
        text_message,
        state,
    )
    orchestrator.from_role = Role.ENV
    orchestrator.to_role = Role.AGENT

    orchestrator.step()

    assert orchestrator.done is True
    assert orchestrator.termination_reason == TerminationReason.AGENT_ERROR


def test_solo_mode_empty_first_message_terminates_with_agent_error(
    domain_name: str,
    dummy_user: DummyUser,
    solo_agent: LLMSoloAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that an empty first solo-mode turn is charged to the agent."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=dummy_user,
        agent=solo_agent,
        environment=get_environment(solo_mode=True),
        task=base_task,
        solo_mode=True,
    )
    empty_message = AssistantMessage(
        role="assistant",
        content=None,
        cost=0.0,
        raw_data={"choices": [{"finish_reason": "tool_calls"}]},
    )
    orchestrator.agent.generate_next_message = lambda message, state: (
        empty_message,
        state,
    )

    orchestrator.initialize()

    assert orchestrator.done is True
    assert orchestrator.termination_reason == TerminationReason.AGENT_ERROR
    assert orchestrator.trajectory == [empty_message]
    assert orchestrator.to_role == Role.USER


def test_solo_mode_content_filter_first_message_is_left_to_the_caller(
    domain_name: str,
    dummy_user: DummyUser,
    solo_agent: LLMSoloAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that the first solo-mode turn goes through the same provider-fault gate."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=dummy_user,
        agent=solo_agent,
        environment=get_environment(solo_mode=True),
        task=base_task,
        solo_mode=True,
    )
    filtered_message = AssistantMessage(
        role="assistant",
        content=None,
        cost=0.0,
        raw_data={"choices": [{"finish_reason": "content_filter"}]},
    )
    orchestrator.agent.generate_next_message = lambda message, state: (
        filtered_message,
        state,
    )

    with pytest.raises(AgentError):
        orchestrator.initialize()

    assert orchestrator.done is False
    assert orchestrator.termination_reason is None


def test_run_with_empty_agent_message_scores_zero(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that an empty agent message ends run() with AGENT_ERROR and a reward of 0."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
        max_steps=10,
    )
    # Both participants must be mocked: the fixtures point at a live model.
    orchestrator.user.generate_next_message = lambda message, state: (
        UserMessage(role="user", content="I need help with my task.", cost=0.0),
        state,
    )
    orchestrator.agent.generate_next_message = lambda message, state: (
        AssistantMessage(role="assistant", content=None, cost=0.0),
        state,
    )

    simulation_run = orchestrator.run()

    assert simulation_run.termination_reason == TerminationReason.AGENT_ERROR
    assert len(simulation_run.messages) == 3
    reward_info = evaluate_simulation(
        simulation=simulation_run,
        task=base_task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain_name,
    )
    assert reward_info.reward == 0.0


def test_run_with_agent_error_before_a_message_scores_zero(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that run() finalizes when the agent fails before producing any message."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
        max_steps=10,
    )

    def raise_agent_error(message, state):
        raise AgentError("Model returned invalid JSON arguments for tool get_task")

    orchestrator.user.generate_next_message = lambda message, state: (
        UserMessage(role="user", content="I need help with my task.", cost=0.0),
        state,
    )
    orchestrator.agent.generate_next_message = raise_agent_error

    simulation_run = orchestrator.run()

    assert simulation_run.termination_reason == TerminationReason.AGENT_ERROR
    assert len(simulation_run.messages) == 2
    reward_info = evaluate_simulation(
        simulation=simulation_run,
        task=base_task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain_name,
    )
    assert reward_info.reward == 0.0


def test_run_solo_mode_agent_error_on_the_first_turn_is_charged(
    domain_name: str,
    dummy_user: DummyUser,
    solo_agent: LLMSoloAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that a first solo turn failing before any message ends run() with AGENT_ERROR."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=dummy_user,
        agent=solo_agent,
        environment=get_environment(solo_mode=True),
        task=base_task,
        solo_mode=True,
        validate_communication=True,
    )

    def raise_agent_error(message, state):
        raise AgentError("Model returned invalid JSON arguments for tool get_task")

    orchestrator.agent.generate_next_message = raise_agent_error

    # initialize() runs outside run()'s try/finally, so a communication check on the charged
    # turn would raise on the still-unset from_role and hand the episode back for a re-roll.
    simulation_run = orchestrator.run()

    assert simulation_run.termination_reason == TerminationReason.AGENT_ERROR
    assert simulation_run.messages == []


def test_empty_user_message_still_raises(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that an empty user message remains an error, since it is an apparatus failure."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    orchestrator.user.generate_next_message = lambda message, state: (
        UserMessage(role="user", content=None, cost=0.0),
        state,
    )

    with pytest.raises(
        ValueError, match="UserMessage must have either content or tool_calls"
    ):
        orchestrator.step()


def test_agent_generation_error_still_raises(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that only a protocol violation is charged to the agent.

    A ValueError raised while generating comes from the harness or the
    configuration rather than from the model's output, so it has to reach the
    caller as a retry. Only `validate()` failures become AgentError.
    """
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    def raise_value_error(message, state):
        raise ValueError("message history is malformed")

    orchestrator.agent.generate_next_message = raise_value_error
    # Hand the turn to the agent.
    orchestrator.from_role = Role.USER
    orchestrator.to_role = Role.AGENT

    with pytest.raises(ValueError, match="message history is malformed"):
        orchestrator.step()

    assert orchestrator.done is False
    assert orchestrator.termination_reason is None


def test_validate_communication_allows_valid_messages(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test that valid messages pass through when validation is enabled."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
        validate_communication=True,
    )
    orchestrator.initialize()

    # Should initialize successfully with valid message
    assert orchestrator.done is False
    assert orchestrator.termination_reason is None
