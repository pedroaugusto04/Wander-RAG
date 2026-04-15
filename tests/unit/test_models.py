"""Tests for domain models."""


from src.core.models import (
    ChannelType,
    ConversationContext,
    IncomingMessage,
    MessageRole,
    OutgoingMessage,
    RetrievedChunk,
)


class TestIncomingMessage:
    def test_create_with_defaults(self) -> None:
        msg = IncomingMessage(
            channel=ChannelType.TELEGRAM,
            channel_user_id="123",
            channel_chat_id="456",
            text="Hello",
        )
        assert msg.text == "Hello"
        assert msg.channel == ChannelType.TELEGRAM
        assert msg.channel_user_id == "123"
        assert msg.id  # auto-generated UUID
        assert msg.timestamp  # auto-generated
        assert msg.metadata == {}
        assert msg.attachments == []

    def test_create_with_metadata(self) -> None:
        msg = IncomingMessage(
            channel=ChannelType.WHATSAPP,
            channel_user_id="user1",
            channel_chat_id="chat1",
            text="Test",
            metadata={"key": "value"},
        )
        assert msg.metadata == {"key": "value"}
        assert msg.channel == ChannelType.WHATSAPP


class TestOutgoingMessage:
    def test_create_basic(self) -> None:
        msg = OutgoingMessage(text="Response", channel_chat_id="456")
        assert msg.text == "Response"
        assert msg.reply_to_message_id is None

    def test_create_with_reply(self) -> None:
        msg = OutgoingMessage(
            text="Reply",
            channel_chat_id="456",
            reply_to_message_id="789",
        )
        assert msg.reply_to_message_id == "789"


class TestConversationContext:
    def test_create_session(self) -> None:
        ctx = ConversationContext(
            session_id="test-session",
            channel=ChannelType.TELEGRAM,
            channel_user_id="user1",
        )
        assert ctx.session_id == "test-session"
        assert ctx.history == []

    def test_add_turn(self) -> None:
        ctx = ConversationContext(
            session_id="s1",
            channel=ChannelType.TELEGRAM,
            channel_user_id="u1",
        )
        ctx.add_turn(MessageRole.USER, "Hello")
        ctx.add_turn(MessageRole.ASSISTANT, "Hi there!")

        assert len(ctx.history) == 2
        assert ctx.history[0] == {"role": "user", "content": "Hello"}
        assert ctx.history[1] == {"role": "assistant", "content": "Hi there!"}

    def test_get_recent_history(self) -> None:
        ctx = ConversationContext(
            session_id="s1",
            channel=ChannelType.TELEGRAM,
            channel_user_id="u1",
        )
        for i in range(20):
            ctx.add_turn(MessageRole.USER, f"Message {i}")

        recent = ctx.get_recent_history(max_turns=5)
        assert len(recent) == 5
        assert recent[0]["content"] == "Message 15"

    def test_add_turn_updates_activity(self) -> None:
        ctx = ConversationContext(
            session_id="s1",
            channel=ChannelType.TELEGRAM,
            channel_user_id="u1",
        )
        original_time = ctx.last_activity
        ctx.add_turn(MessageRole.USER, "Test")
        assert ctx.last_activity >= original_time


class TestRetrievedChunk:
    def test_create(self) -> None:
        chunk = RetrievedChunk(
            content="Some text",
            score=0.85,
            metadata={"document_title": "Manual"},
        )
        assert chunk.content == "Some text"
        assert chunk.score == 0.85
        assert chunk.metadata["document_title"] == "Manual"


class TestChannelType:
    def test_values(self) -> None:
        assert ChannelType.TELEGRAM.value == "telegram"
        assert ChannelType.WHATSAPP.value == "whatsapp"
        assert ChannelType.WEB.value == "web"
