import pytest
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    ModelResponse,
)
from litellm.types.utils import Message as LiteLLMMessage

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import ToolCallArgumentsError, generate


@pytest.fixture
def model() -> str:
    return "gpt-4o-mini"


@pytest.fixture
def messages() -> list[Message]:
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="What is the capital of the moon?"),
    ]
    return messages


@pytest.fixture
def tool() -> Tool:
    def calculate_square(x: int) -> int:
        """Calculate the square of a number.
            Args:
            x (int): The number to calculate the square of.
        Returns:
            int: The square of the number.
        """
        return x * x

    return as_tool(calculate_square)


@pytest.fixture
def tool_call_messages() -> list[Message]:
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(
            role="user",
            content="What is the square of 5? Just give me the number, no explanation.",
        ),
    ]
    return messages


def test_generate_no_tool_call(model: str, messages: list[Message]):
    response = generate(model, messages)
    assert isinstance(response, AssistantMessage)
    assert response.content is not None


def test_generate_tool_call(model: str, tool_call_messages: list[Message], tool: Tool):
    response = generate(model, tool_call_messages, tools=[tool])
    assert isinstance(response, AssistantMessage)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "calculate_square"
    assert response.tool_calls[0].arguments == {"x": 5}
    follow_up_messages = [
        response,
        ToolMessage(role="tool", id=response.tool_calls[0].id, content="25"),
    ]
    response = generate(
        model,
        tool_call_messages + follow_up_messages,
        tools=[tool],
    )
    assert isinstance(response, AssistantMessage)
    assert response.tool_calls is None
    assert response.content == "25"


def test_generate_raises_on_invalid_tool_call_arguments(
    model: str,
    tool_call_messages: list[Message],
    tool: Tool,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that unparseable tool call arguments raise ToolCallArgumentsError."""
    response = ModelResponse(
        model=model,
        choices=[
            Choices(
                index=0,
                finish_reason="tool_calls",
                message=LiteLLMMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(
                                name="calculate_square", arguments="{not json"
                            ),
                        )
                    ],
                ),
            )
        ],
    )
    monkeypatch.setattr("tau2.utils.llm_utils.completion", lambda **kwargs: response)

    # A dedicated type, so the agent layer can charge this to the model without also charging
    # the JSONDecodeError litellm raises on a malformed response body.
    with pytest.raises(ToolCallArgumentsError):
        generate(model, tool_call_messages, tools=[tool])
