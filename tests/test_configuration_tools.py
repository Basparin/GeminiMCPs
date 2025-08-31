import pytest
import os
from unittest.mock import patch, MagicMock
from codesage_mcp.tools.configuration import (
    configure_api_key_tool,
    get_configuration_tool,
    get_cache_statistics_tool,
)


class TestConfigureApiKeyTool:
    """Test cases for configure_api_key_tool function."""

    def test_configure_api_key_tool_valid_groq(self):
        """Test configuring a valid Groq API key."""
        with patch.dict(os.environ, {}, clear=True):
            result = configure_api_key_tool("groq", "test_groq_key")

            assert "message" in result
            assert "groq" in result["message"]
            assert "updated successfully" in result["message"]
            assert os.environ.get("GROQ_API_KEY") == "test_groq_key"

    def test_configure_api_key_tool_valid_openrouter(self):
        """Test configuring a valid OpenRouter API key."""
        with patch.dict(os.environ, {}, clear=True):
            result = configure_api_key_tool("openrouter", "test_openrouter_key")

            assert "message" in result
            assert "openrouter" in result["message"]
            assert "updated successfully" in result["message"]
            assert os.environ.get("OPENROUTER_API_KEY") == "test_openrouter_key"

    def test_configure_api_key_tool_valid_google(self):
        """Test configuring a valid Google API key."""
        with patch.dict(os.environ, {}, clear=True):
            result = configure_api_key_tool("google", "test_google_key")

            assert "message" in result
            assert "google" in result["message"]
            assert "updated successfully" in result["message"]
            assert os.environ.get("GOOGLE_API_KEY") == "test_google_key"

    def test_configure_api_key_tool_invalid_provider(self):
        """Test configuring API key with invalid provider."""
        result = configure_api_key_tool("invalid_provider", "test_key")

        assert "error" in result
        assert result["error"]["code"] == "INVALID_PROVIDER"
        assert "Unsupported LLM provider" in result["error"]["message"]

    def test_configure_api_key_tool_empty_key(self):
        """Test configuring API key with empty key."""
        result = configure_api_key_tool("groq", "")

        assert "error" in result
        assert result["error"]["code"] == "INVALID_API_KEY"
        assert "API key cannot be empty" in result["error"]["message"]

    def test_configure_api_key_tool_whitespace_key(self):
        """Test configuring API key with whitespace-only key."""
        result = configure_api_key_tool("groq", "   ")

        assert "error" in result
        assert result["error"]["code"] == "INVALID_API_KEY"
        assert "API key cannot be empty" in result["error"]["message"]


class TestGetConfigurationTool:
    """Test cases for get_configuration_tool function."""

    @patch('codesage_mcp.config.get_configuration_status')
    def test_get_configuration_tool_with_keys(self, mock_get_status):
        """Test getting configuration with API keys set."""
        mock_get_status.return_value = {"status": "configured"}

        with patch('codesage_mcp.config.GROQ_API_KEY', 'sk-groq123'):
            with patch('codesage_mcp.config.OPENROUTER_API_KEY', 'sk-openrouter456'):
                with patch('codesage_mcp.config.GOOGLE_API_KEY', 'sk-google789'):
                    result = get_configuration_tool()

                    assert "message" in result
                    assert "configuration" in result
                    assert result["configuration"]["groq_api_key"] == "sk-g...q123"
                    assert result["configuration"]["openrouter_api_key"] == "sk-o...r456"
                    assert result["configuration"]["google_api_key"] == "sk-g...e789"

    @patch('codesage_mcp.config.get_configuration_status')
    def test_get_configuration_tool_no_keys(self, mock_get_status):
        """Test getting configuration with no API keys set."""
        mock_get_status.return_value = {"status": "not_configured"}

        with patch('codesage_mcp.config.GROQ_API_KEY', ''):
            with patch('codesage_mcp.config.OPENROUTER_API_KEY', ''):
                with patch('codesage_mcp.config.GOOGLE_API_KEY', ''):
                    result = get_configuration_tool()

                    assert "message" in result
                    assert "configuration" in result
                    assert result["configuration"]["groq_api_key"] == "Not set"
                    assert result["configuration"]["openrouter_api_key"] == "Not set"
                    assert result["configuration"]["google_api_key"] == "Not set"

    @patch('codesage_mcp.config.get_configuration_status')
    def test_get_configuration_tool_short_keys(self, mock_get_status):
        """Test getting configuration with short API keys."""
        mock_get_status.return_value = {"status": "configured"}

        with patch('codesage_mcp.config.GROQ_API_KEY', '123'):
            with patch('codesage_mcp.config.OPENROUTER_API_KEY', '4567'):
                with patch('codesage_mcp.config.GOOGLE_API_KEY', '89'):
                    result = get_configuration_tool()

                    assert result["configuration"]["groq_api_key"] == "***"
                    assert result["configuration"]["openrouter_api_key"] == "****"
                    assert result["configuration"]["google_api_key"] == "**"


class TestGetCacheStatisticsTool:
    """Test cases for get_cache_statistics_tool function."""

    @patch('codesage_mcp.tools.configuration.ENABLE_CACHING', True)
    @patch('codesage_mcp.tools.configuration.get_cache_instance')
    def test_get_cache_statistics_tool_enabled(self, mock_get_cache):
        """Test getting cache statistics when caching is enabled."""
        mock_cache = MagicMock()
        mock_cache.get_comprehensive_stats.return_value = {"hits": 100, "misses": 20}
        mock_get_cache.return_value = mock_cache

        result = get_cache_statistics_tool()

        assert "message" in result
        assert result["caching_enabled"] is True
        assert "statistics" in result
        assert result["statistics"]["hits"] == 100
        assert result["statistics"]["misses"] == 20

    @patch('codesage_mcp.tools.configuration.ENABLE_CACHING', False)
    def test_get_cache_statistics_tool_disabled(self):
        """Test getting cache statistics when caching is disabled."""
        result = get_cache_statistics_tool()

        assert "message" in result
        assert result["caching_enabled"] is False
        assert "Caching is disabled" in result["message"]

    @patch('codesage_mcp.tools.configuration.ENABLE_CACHING', True)
    @patch('codesage_mcp.tools.configuration.get_cache_instance')
    def test_get_cache_statistics_tool_error(self, mock_get_cache):
        """Test getting cache statistics when an error occurs."""
        mock_get_cache.side_effect = Exception("Cache connection failed")

        result = get_cache_statistics_tool()

        assert "code" in result
        assert result["code"] == -32006  # CACHE_ERROR code
        assert "Failed to retrieve cache statistics" in result["message"]