# Logging Configuration

The SDK provides a centralized logging system that can be configured via environment variables.

## Global Log Level

Set the log level for all SDK modules:

```bash
# Set global log level (default: WARNING)
export PAI_AGENT_LOG_LEVEL=DEBUG  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Module-Specific Log Levels

Override log levels for specific modules. The environment variable format is:

```
PAI_AGENT_LOG_LEVEL_<MODULE_PATH>
```

Where `<MODULE_PATH>` is the module path with dots replaced by underscores and converted to uppercase.

**Examples:**

```bash
# Enable debug logging for browser_use module only
export PAI_AGENT_LOG_LEVEL_TOOLSETS_BROWSER_USE=DEBUG

# Enable debug logging for all toolsets
export PAI_AGENT_LOG_LEVEL_TOOLSETS=DEBUG

# Enable info logging for sandbox module
export PAI_AGENT_LOG_LEVEL_SANDBOX=INFO

# Combine global and module-specific settings
export PAI_AGENT_LOG_LEVEL=WARNING           # Default to WARNING
export PAI_AGENT_LOG_LEVEL_TOOLSETS_BROWSER_USE=DEBUG  # But DEBUG for browser_use
```

## Module Hierarchy

Module-specific settings are checked from most specific to least specific:

| Module Path                        | Environment Variable                                   |
| ---------------------------------- | ------------------------------------------------------ |
| `toolsets.browser_use.tools.query` | `PAI_AGENT_LOG_LEVEL_TOOLSETS_BROWSER_USE_TOOLS_QUERY` |
| `toolsets.browser_use.tools`       | `PAI_AGENT_LOG_LEVEL_TOOLSETS_BROWSER_USE_TOOLS`       |
| `toolsets.browser_use`             | `PAI_AGENT_LOG_LEVEL_TOOLSETS_BROWSER_USE`             |
| `toolsets`                         | `PAI_AGENT_LOG_LEVEL_TOOLSETS`                         |

## Programmatic Usage

```python
from pai_agent_sdk._logger import get_logger

# Get a logger for your module
logger = get_logger(__name__)

# Or get a logger with a custom name
logger = get_logger("my_custom_module")

# Log messages
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.exception("Exception with traceback")  # Use in except blocks
```

## Output Format

Logs are written to stderr with the following format:

**Terminal (with color support):**

```
2026-01-05 22:00:00 | DEBUG    | toolsets.browser_use:function_name:42 - Message
```

**Non-terminal (plain text):**

```
2026-01-05 22:00:00 | DEBUG    | pai_agent_sdk.toolsets.browser_use:function_name:42 - Message
```

## Best Practices

1. **Use `get_logger(__name__)`** in each module to get a properly namespaced logger
2. **Use `logger.exception()`** in except blocks - it automatically includes the traceback
3. **Don't include the exception in the message** when using `logger.exception()`:
   ```python
   # Good
   except Exception:
       logger.exception("Failed to process request")

   # Bad (redundant)
   except Exception as e:
       logger.exception(f"Failed to process request: {e}")
   ```
4. **Use appropriate log levels**:
   - `DEBUG`: Detailed diagnostic information
   - `INFO`: General operational events
   - `WARNING`: Unexpected but handled situations
   - `ERROR`: Errors that prevent normal operation
   - `CRITICAL`: Severe errors that may cause application failure
