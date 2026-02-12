# Gradio GUI

AgribotTools includes a **Gradio-based graphical interface** for running format conversions without writing code.

---

## Launching the GUI

```bash
python gui.py
```

The GUI will start a local web server, accessible at **[http://localhost:7860](http://localhost:7860)** by default.

### Server Options

The GUI launches with the following default settings:

| Option | Default |
|--------|---------|
| Server address | `0.0.0.0` (all interfaces) |
| Port | `7860` |
| Public sharing | Disabled |

---

## Features

The GUI provides a unified interface for:

- **Selecting source and target formats** — dropdown menus populated from the format registry.
- **Browsing input/output directories** — file pickers for source and target paths.
- **Running conversions** — one-click conversion with real-time progress feedback.
- **Format validation** — automatic validation of the source dataset before conversion.
- **Error reporting** — clear error messages displayed in the interface.

---

## Supported Conversions

The GUI dynamically discovers all registered conversions. When you select a source format, only the valid target formats are shown in the target dropdown.

This is powered by the same `FormatRegistry` used by the Python API. See the [API Reference](../api/formats.md) for details.

---

## Architecture

The GUI is implemented across three modules:

| Module | Responsibility |
|--------|---------------|
| `gui.py` | Main Gradio Blocks interface creation and launch. |
| `src/ui/formats.py` | Format dropdown population from the registry. |
| `src/ui/callbacks.py` | Conversion execution callbacks with progress reporting. |

---

## Dev Container Support

The repository includes a `.devcontainer/` configuration for running AgribotTools (including the GUI) inside a Docker-based development container. This is useful for consistent environments and remote development.
