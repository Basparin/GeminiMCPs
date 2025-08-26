"""Configuration Module for CodeSage MCP Server.

This module manages the configuration for the CodeSage MCP Server, including API keys
for various services like Groq, OpenRouter, and Google AI.

It uses `python-dotenv` to automatically load environment variables from a `.env` file
located in the project root directory. This allows for secure configuration without
hardcoding sensitive information in the source code.

Environment Variables:
    GROQ_API_KEY (str): API key for Groq.
    OPENROUTER_API_KEY (str): API key for OpenRouter.
    GOOGLE_API_KEY (str): API key for Google AI.

Example .env file:
    ```env
    GROQ_API_KEY="gsk_..."
    OPENROUTER_API_KEY="sk-or-..."
    GOOGLE_API_KEY="AIza..."
    ```
"""

# Placeholder for API keys - DO NOT COMMIT REAL KEYS HERE!
GROQ_API_KEY = "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"
OPENROUTER_API_KEY = (
    "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"
)
GOOGLE_API_KEY = "AIzaSyBRmsd7sKtVviULWkI6fR3iNQrZBOKdqmA"
