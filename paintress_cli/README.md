# Paintress CLI

TUI reference implementation for [pai-agent-sdk](https://github.com/youware-labs/pai-agent-sdk).

## Usage

With uvx, run:

```bash
uvx paintress-cli
```

Or to install paintress-cli globally with uv, run:

```bash
uv tool install paintress-cli
...
paintress
```

Or with pip, run:

```bash
pip install paintress-cli
...
paintress
```

Or run as a module:

```bash
python -m paintress_cli
```

## Development

This package is part of the pai-agent-sdk monorepo. To develop locally:

```bash
cd pai-agent-sdk
uv sync --all-packages
```

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.
