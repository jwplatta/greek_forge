# Logging

Greek Forge uses a singleton logger pattern to ensure consistent logging configuration across the entire application.

## Quick Start

```python
from src.utils.logger import get_logger

logger = get_logger()

logger.info("Training started")
logger.warning("Low sample count detected")
logger.error("Model loading failed")
```

## Features

- **Singleton pattern**: All modules share the same logger configuration
- **Environment variable support**: Configure via `LOG_LEVEL` and `LOG_FILE` env vars
- **Console + file logging**: Logs to STDOUT by default, optionally to file
- **Timestamp formatting**: `YYYY-MM-DD HH:MM:SS - name - level - message`

## Configuration

### Default (Console Only)

```python
from src.utils.logger import get_logger

logger = get_logger()
logger.info("This goes to STDOUT")
```

### With File Logging

```python
from src.utils.logger import setup_logger
from pathlib import Path

logger = setup_logger(log_file=Path("logs/app.log"))
logger.info("This goes to both console and file")
```

### Using Environment Variables

Set environment variables in your `.env` file:

```bash
LOG_LEVEL=DEBUG
LOG_FILE=logs/greek_forge.log
```

Then the logger will automatically use these settings:

```python
from src.utils.logger import get_logger

logger = get_logger()
logger.debug("This uses env var configuration")
```

### Changing Log Level at Runtime

```python
import logging
from src.utils.logger import setup_logger

# Set to DEBUG for verbose output
logger = setup_logger(level=logging.DEBUG)
logger.debug("This is now visible")

# Set back to INFO for production
logger = setup_logger(level=logging.INFO)
logger.debug("This won't show")
```

## Log Levels

From most to least verbose:

- `logging.DEBUG` - Detailed diagnostic information
- `logging.INFO` - General informational messages (default)
- `logging.WARNING` - Warning messages
- `logging.ERROR` - Error messages
- `logging.CRITICAL` - Critical error messages

## Singleton Behavior

All modules get the same logger instance:

```python
# In src/models/trainer.py
from src.utils.logger import get_logger
logger = get_logger()
logger.info("Training started")

# In src/api/predictor.py
from src.utils.logger import get_logger
logger = get_logger()
logger.info("Loading model")

# Both use the same logger instance and configuration!
# logger from trainer.py is logger from predictor.py → True
```

## Best Practices

1. **Import at module level**:
   ```python
   from src.utils.logger import get_logger
   logger = get_logger()
   ```

2. **Use appropriate log levels**:
   - `debug()` for diagnostic info during development
   - `info()` for general operational messages
   - `warning()` for potential issues
   - `error()` for errors that need attention
   - `critical()` for serious errors

3. **Use f-strings for formatting**:
   ```python
   logger.info(f"Loaded {len(df)} samples")
   ```

4. **Avoid logging sensitive data**:
   ```python
   # DON'T
   logger.info(f"API key: {api_key}")

   # DO
   logger.info("API key configured successfully")
   ```

## Production Deployment

For production, set these environment variables:

```bash
LOG_LEVEL=INFO
LOG_FILE=logs/greek_forge.log
```

This will:
- Log INFO level and above (filtering out DEBUG)
- Write logs to both console and file
- Use timestamps for log rotation and debugging
