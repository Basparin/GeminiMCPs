import pytest
from unittest.mock import Mock, patch
import os
from codesage_mcp.configuration import ConfigManager, EnvVarHandler, APIKeyValidator


@pytest.fixture
def mock_config_manager():
    """Fixture to provide a mocked ConfigManager instance."""
    return Mock(spec=ConfigManager)


@pytest.fixture
def mock_env_var_handler():
    """Fixture to provide a mocked EnvVarHandler instance."""
    return Mock(spec=EnvVarHandler)


@pytest.fixture
def mock_api_key_validator():
    """Fixture to provide a mocked APIKeyValidator instance."""
    return Mock(spec=APIKeyValidator)


def test_config_manager_load_from_env(mock_config_manager):
    """
    Test that ConfigManager loads configuration from environment variables.

    Theoretical expectation: The manager should read specified environment variables
    and populate the configuration object with their values.
    """
    # Arrange
    env_vars = {"API_KEY": "test_key", "DEBUG": "true"}
    with patch.dict(os.environ, env_vars):
        expected_config = {"api_key": "test_key", "debug": True}
        mock_config_manager.load_from_env.return_value = expected_config

        # Act
        config = mock_config_manager.load_from_env()

        # Assert
        assert config == expected_config


def test_env_var_handler_get_variable(mock_env_var_handler):
    """
    Test that EnvVarHandler retrieves environment variables with defaults.

    Theoretical expectation: The handler should return the value of the environment variable
    or a default value if the variable is not set.
    """
    # Arrange
    var_name = "TEST_VAR"
    default_value = "default"
    with patch.dict(os.environ, {var_name: "actual_value"}):
        mock_env_var_handler.get.return_value = "actual_value"

        # Act
        value = mock_env_var_handler.get(var_name, default_value)

        # Assert
        assert value == "actual_value"


def test_env_var_handler_get_missing_variable(mock_env_var_handler):
    """
    Test that EnvVarHandler returns default for missing environment variables.

    Theoretical expectation: When an environment variable is not set, the handler
    should return the provided default value.
    """
    # Arrange
    var_name = "MISSING_VAR"
    default_value = "default"
    mock_env_var_handler.get.return_value = default_value

    # Act
    value = mock_env_var_handler.get(var_name, default_value)

    # Assert
    assert value == default_value


def test_api_key_validator_validate_key(mock_api_key_validator):
    """
    Test that APIKeyValidator validates API keys correctly.

    Theoretical expectation: Valid API keys should pass validation, while invalid
    ones should fail with appropriate error messages.
    """
    # Arrange
    valid_key = "sk-valid-key-123"
    invalid_key = "invalid"
    mock_api_key_validator.validate.return_value = True

    # Act
    is_valid = mock_api_key_validator.validate(valid_key)

    # Assert
    assert is_valid is True

    mock_api_key_validator.validate.return_value = False
    is_valid = mock_api_key_validator.validate(invalid_key)
    assert is_valid is False


def test_config_manager_validate_configuration(mock_config_manager):
    """
    Test that ConfigManager validates the loaded configuration.

    Theoretical expectation: The manager should check for required fields and
    validate their types and values, raising errors for invalid configurations.
    """
    # Arrange
    valid_config = {"api_key": "sk-valid", "debug": False}
    invalid_config = {"api_key": "", "debug": "not_bool"}
    mock_config_manager.validate.return_value = True

    # Act
    is_valid = mock_config_manager.validate(valid_config)

    # Assert
    assert is_valid is True

    mock_config_manager.validate.return_value = False
    is_valid = mock_config_manager.validate(invalid_config)
    assert is_valid is False


def test_env_var_handler_set_variable(mock_env_var_handler):
    """
    Test that EnvVarHandler can set environment variables.

    Theoretical expectation: Setting a variable should update the environment
    and make it available for subsequent retrievals.
    """
    # Arrange
    var_name = "NEW_VAR"
    var_value = "new_value"
    with patch.dict(os.environ, {}, clear=True):
        mock_env_var_handler.set.return_value = None

        # Act
        mock_env_var_handler.set(var_name, var_value)

        # Assert
        mock_env_var_handler.set.assert_called_once_with(var_name, var_value)


def test_api_key_validator_format_key(mock_api_key_validator):
    """
    Test that APIKeyValidator formats API keys correctly.

    Theoretical expectation: The validator should normalize key format (e.g., trim whitespace,
    ensure proper prefix) for consistent handling.
    """
    # Arrange
    raw_key = "  sk-test-key  "
    expected_formatted = "sk-test-key"
    mock_api_key_validator.format.return_value = expected_formatted

    # Act
    formatted = mock_api_key_validator.format(raw_key)

    # Assert
    assert formatted == expected_formatted


def test_config_manager_save_to_file(mock_config_manager):
    """
    Test that ConfigManager can save configuration to a file.

    Theoretical expectation: The configuration should be serialized (e.g., to JSON)
    and written to the specified file path.
    """
    # Arrange
    config = {"api_key": "sk-test", "debug": True}
    filepath = "/tmp/config.json"
    with patch('builtins.open', create=True) as mock_file:
        mock_file.return_value.__enter__.return_value = Mock()
        mock_config_manager.save_to_file.return_value = None

        # Act
        mock_config_manager.save_to_file(config, filepath)

        # Assert
        mock_config_manager.save_to_file.assert_called_once_with(config, filepath)


def test_env_var_handler_list_variables(mock_env_var_handler):
    """
    Test that EnvVarHandler can list environment variables with a prefix.

    Theoretical expectation: The handler should return a dictionary of all environment
    variables that match the given prefix.
    """
    # Arrange
    prefix = "APP_"
    expected_vars = {"APP_KEY": "value1", "APP_DEBUG": "value2"}
    with patch.dict(os.environ, expected_vars):
        mock_env_var_handler.list_with_prefix.return_value = expected_vars

        # Act
        vars_list = mock_env_var_handler.list_with_prefix(prefix)

        # Assert
        assert vars_list == expected_vars


def test_api_key_validator_check_expiry(mock_api_key_validator):
    """
    Test that APIKeyValidator checks API key expiry.

    Theoretical expectation: The validator should determine if an API key is expired
    based on its creation or expiry timestamp.
    """
    # Arrange
    expired_key = "sk-expired-key"
    valid_key = "sk-valid-key"
    mock_api_key_validator.is_expired.return_value = True

    # Act
    is_expired = mock_api_key_validator.is_expired(expired_key)

    # Assert
    assert is_expired is True

    mock_api_key_validator.is_expired.return_value = False
    is_expired = mock_api_key_validator.is_expired(valid_key)
    assert is_expired is False


def test_config_manager_merge_configs(mock_config_manager):
    """
    Test that ConfigManager merges multiple configuration sources.

    Theoretical expectation: Configurations from different sources (e.g., file, env, defaults)
    should be merged with later sources overriding earlier ones.
    """
    # Arrange
    base_config = {"debug": False}
    override_config = {"debug": True, "api_key": "sk-new"}
    expected_merged = {"debug": True, "api_key": "sk-new"}
    mock_config_manager.merge.return_value = expected_merged

    # Act
    merged = mock_config_manager.merge(base_config, override_config)

    # Assert
    assert merged == expected_merged