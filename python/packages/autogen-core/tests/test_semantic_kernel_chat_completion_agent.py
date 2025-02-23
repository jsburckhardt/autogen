import pytest
from unittest.mock import AsyncMock, MagicMock
from autogen.core.agents.semantic_kernel_chat_completion_agent import SemanticKernelChatCompletionAgent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments

class TestSemanticKernelChatCompletionAgent:

    @pytest.mark.asyncio
    async def test_generate_reply_async(self):
        agent = SemanticKernelChatCompletionAgent(name="test_agent")
        agent.build_chat_history = MagicMock(return_value=ChatHistory())
        agent._get_chat_completion_service_and_settings = AsyncMock(return_value=(AsyncMock(), MagicMock()))
        agent._get_chat_completion_service_and_settings.return_value[0].complete_chat_async = AsyncMock(return_value=AsyncMock(__aiter__=lambda s: iter([ChatMessageContent(role=AuthorRole.USER, content="test_reply")])))

        result = await agent.generate_reply_async(messages=[], kernel=MagicMock(), arguments=KernelArguments())
        assert result.content == "test_reply"

    def test_build_chat_history(self):
        agent = SemanticKernelChatCompletionAgent(name="test_agent")
        messages = [ChatMessageContent(role=AuthorRole.USER, content="test_message")]
        history = agent.build_chat_history(messages)
        assert len(history.messages) == 1
        assert history.messages[0].content == "test_message"

    def test_process_message(self):
        agent = SemanticKernelChatCompletionAgent(name="test_agent")
        message = "test_message"
        result = agent.process_message(message)
        assert result.content == "test_message"

        message = {"content": "test_message"}
        result = agent.process_message(message)
        assert result.content == "test_message"

        with pytest.raises(ValueError):
            agent.process_message(123)
