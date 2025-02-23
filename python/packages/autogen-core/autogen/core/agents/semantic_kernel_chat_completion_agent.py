import logging
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any, ClassVar

from semantic_kernel.agents import Agent
from semantic_kernel.agents.channels.agent_channel import AgentChannel
from semantic_kernel.agents.channels.chat_history_channel import ChatHistoryChannel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.const import DEFAULT_SERVICE_NAME
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.exceptions import KernelServiceNotFoundError
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions.kernel_function import TEMPLATE_FORMAT_MAP
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig

class SemanticKernelChatCompletionAgent(Agent):
    def __init__(self, name: str, service_id: str = DEFAULT_SERVICE_NAME):
        super().__init__(name)
        self.service_id = service_id

    async def generate_reply_async(self, messages: list[ChatMessageContent], kernel: "Kernel", arguments: KernelArguments) -> ChatMessageContent:
        chat_history = self.build_chat_history(messages)
        chat_completion_service, settings = await self._get_chat_completion_service_and_settings(kernel, arguments)
        async for message in chat_completion_service.complete_chat_async(chat_history, settings):
            return message

    def build_chat_history(self, messages: list[ChatMessageContent]) -> ChatHistory:
        history = ChatHistory()
        for message in messages:
            history.add_message(message)
        return history

    def process_message(self, message: Any) -> ChatMessageContent:
        if isinstance(message, str):
            return ChatMessageContent(role=AuthorRole.USER, content=message)
        elif isinstance(message, dict):
            return ChatMessageContent(role=AuthorRole.USER, content=message.get("content", ""))
        else:
            raise ValueError("Unsupported message type")

    async def _get_chat_completion_service_and_settings(
        self, kernel: "Kernel", arguments: KernelArguments
    ) -> tuple[ChatCompletionClientBase, PromptExecutionSettings]:
        chat_completion_service, settings = kernel.select_ai_service(arguments=arguments, type=ChatCompletionClientBase)

        if not chat_completion_service:
            raise KernelServiceNotFoundError(f"Chat completion service not found with service_id: {self.service_id}")

        assert isinstance(chat_completion_service, ChatCompletionClientBase)  # nosec
        assert settings is not None  # nosec

        return chat_completion_service, settings
