"""Config flow for OpenAI Conversation integration."""
from __future__ import annotations

import logging
import types
from types import MappingProxyType
from typing import Any

# Transform the original OpenAI to Google GenAI
from google import genai
from google.genai.errors import APIError, ClientError
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    Content,
    FunctionDeclaration,
    GenerateContentConfig,
    GenerateContentResponse,
    HarmCategory,
    Part,
    SafetySetting,
    Schema,
    Tool,
)


from openai._exceptions import APIConnectionError, AuthenticationError
import voluptuous as vol
import yaml

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant

# Imports Home Assistant classes, constants, and helper methods. 
# config_entries is needed for building config flows; 
# selectors are user interface widgets that help create forms in the Home Assistant UI.
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)


from .const import (
    CONF_API_VERSION,
    CONF_ATTACH_USERNAME,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_FUNCTIONS,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_ORGANIZATION,
    CONF_PROMPT,
    CONF_SKIP_AUTHENTICATION,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_USE_TOOLS,
    CONTEXT_TRUNCATE_STRATEGIES,
    DEFAULT_ATTACH_USERNAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONF_BASE_URL,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NAME,
    DEFAULT_PROMPT,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_USE_TOOLS,
    DOMAIN,
)
from .helpers import (
    validate_authentication, 
    validate_authentication_new,
)

# Creates a logger for outputting log messages under the current module name.
_LOGGER = logging.getLogger(__name__)

# Defines a schema for the initial form displayed to users. 
# This includes fields such as API key, an optional name, base_url, etc.
STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_NAME): str,
        vol.Required(CONF_API_KEY): str,
        vol.Optional(CONF_BASE_URL, default=DEFAULT_CONF_BASE_URL): str,
        vol.Optional(CONF_API_VERSION): str,
        vol.Optional(CONF_ORGANIZATION): str,
        vol.Optional(
            CONF_SKIP_AUTHENTICATION, default=DEFAULT_SKIP_AUTHENTICATION
        ): bool,
    }
)

# Dumps a default functions structure (from Python to YAML) so it can be used in the form as the default value.
DEFAULT_CONF_FUNCTIONS_STR = yaml.dump(DEFAULT_CONF_FUNCTIONS, sort_keys=False)

# Defines a read-only dictionary (using MappingProxyType) that holds default option values for prompts, models, function calls, and so forth.
DEFAULT_OPTIONS = types.MappingProxyType(
    {
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_FUNCTIONS: DEFAULT_CONF_FUNCTIONS_STR,
        CONF_ATTACH_USERNAME: DEFAULT_ATTACH_USERNAME,
        CONF_USE_TOOLS: DEFAULT_USE_TOOLS,
        CONF_CONTEXT_THRESHOLD: DEFAULT_CONTEXT_THRESHOLD,
        CONF_CONTEXT_TRUNCATE_STRATEGY: DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    }
)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    api_key = data[CONF_API_KEY]
    base_url = data.get(CONF_BASE_URL)
    api_version = data.get(CONF_API_VERSION)
    organization = data.get(CONF_ORGANIZATION)
    skip_authentication = data.get(CONF_SKIP_AUTHENTICATION)

    if base_url == DEFAULT_CONF_BASE_URL:
        base_url = None
        data.pop(CONF_BASE_URL)

    # Pause the OpenAI validation fuction.
    #await validate_authentication(
    #    hass=hass,
    #    api_key=api_key,
    #    base_url=base_url,
    #    api_version=api_version,
    #    organization=organization,
    #    skip_authentication=skip_authentication,
    #)
    
    # Use the Gen AI validation function
    await validate_authentication_new(
        hass=hass,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        organization=organization,
        skip_authentication=skip_authentication,
    )
    


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for GenAI Conversation."""

    # The version of this config flow’s schema. 
    # If the schema changes in the future, this can be incremented to handle migrations.
    # This is version 1.1 for GenAI
    VERSION = 1.1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors = {}

        try:
            await validate_input(self.hass, user_input)

        # Use the GenAI Error 
        except APIError:
            errors["base"] = "cannot_connect"
        except ClientError:
            errors["base"] = "invalid_auth"
        
        # Block the OpenAI Error
        #except APIConnectionError:
        #    errors["base"] = "cannot_connect"
        #except AuthenticationError:
        #    errors["base"] = "invalid_auth"
        
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            # If validation is successful, it creates a new config entry with the user’s input. 
            # This means the integration is set up in Home Assistant.
            return self.async_create_entry(
                title=user_input.get(CONF_NAME, DEFAULT_NAME), data=user_input
            )

        # If there were errors, it redisplays the form, populating an error message.
        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    # Tells Home Assistant how to set up an options flow for users who want to modify options after the initial setup.
    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return OptionsFlow(config_entry)

# A custom class for presenting a separate form that configures advanced options.
class OptionsFlow(config_entries.OptionsFlow):
    """GenAI config flow options handler."""

    # Stores the current config entry so the options flow knows the existing configuration.
    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    # The primary method for this options flow. 
    # If the user has provided new input, create or update the entry. Otherwise, show the form.
    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(
                title=user_input.get(CONF_NAME, DEFAULT_NAME), data=user_input
            )
        # If no data has been submitted yet, the code generates a schema for the form (`genai_config_option_schema`) and shows it.
        schema = self.genai_config_option_schema(self.config_entry.options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )

    # A helper method that builds a schema dictionary for the options form. 
    # If there are no existing options, it falls back to `DEFAULT_OPTIONS`.
    def genai_config_option_schema(self, options: MappingProxyType[str, Any]) -> dict:
        """Return a schema for GenAI completion options."""
        if not options:
            options = DEFAULT_OPTIONS

        return {
            vol.Optional(
                CONF_PROMPT,
                description={"suggested_value": options[CONF_PROMPT]},
                default=DEFAULT_PROMPT,
            ): TemplateSelector(),
            vol.Optional(
                CONF_CHAT_MODEL,
                description={
                    # New key in HA 2023.4
                    "suggested_value": options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
                },
                default=DEFAULT_CHAT_MODEL,
            ): str,
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options[CONF_MAX_TOKENS]},
                default=DEFAULT_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": options[CONF_TOP_P]},
                default=DEFAULT_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options[CONF_TEMPERATURE]},
                default=DEFAULT_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                description={
                    "suggested_value": options[CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION]
                },
                default=DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            ): int,
            vol.Optional(
                CONF_FUNCTIONS,
                description={"suggested_value": options.get(CONF_FUNCTIONS)},
                default=DEFAULT_CONF_FUNCTIONS_STR,
            ): TemplateSelector(),
            vol.Optional(
                CONF_ATTACH_USERNAME,
                description={"suggested_value": options.get(CONF_ATTACH_USERNAME)},
                default=DEFAULT_ATTACH_USERNAME,
            ): BooleanSelector(),
            vol.Optional(
                CONF_USE_TOOLS,
                description={"suggested_value": options.get(CONF_USE_TOOLS)},
                default=DEFAULT_USE_TOOLS,
            ): BooleanSelector(),
            vol.Optional(
                CONF_CONTEXT_THRESHOLD,
                description={"suggested_value": options.get(CONF_CONTEXT_THRESHOLD)},
                default=DEFAULT_CONTEXT_THRESHOLD,
            ): int,
            vol.Optional(
                CONF_CONTEXT_TRUNCATE_STRATEGY,
                description={
                    "suggested_value": options.get(CONF_CONTEXT_TRUNCATE_STRATEGY)
                },
                default=DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
            ): SelectSelector(
                # A select dropdown (`SelectSelector`) for the context truncation strategy. 
                # The possible choices are taken from `CONTEXT_TRUNCATE_STRATEGIES`.
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=strategy["key"], label=strategy["label"])
                        for strategy in CONTEXT_TRUNCATE_STRATEGIES
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
        }
