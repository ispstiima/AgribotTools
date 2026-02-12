# Getting Started

## Prerequisites

- **Python** >= 3.8
- **pip** (Python package manager)
- (Optional) **Label Studio** for label-studio format conversions

---

## Installation

We recommend using a virtual environment:

=== "Linux / macOS"

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

=== "Windows"

    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

Then clone and install:

```bash
git clone https://github.com/LambdaLekter/AgribotTools.git
cd AgribotTools
pip install -e .
```

This will install all required dependencies:

- `opencv-python`
- `numpy`
- `python-dotenv`
- `tqdm`
- `PyYAML`
- `pillow`
- `requests`

---

## Environment Variables

Some conversions involving **Label Studio** require environment variables to be set.

!!! warning "Required for Label Studio conversions"
    Make sure to configure the following environment variables on the system that runs AgribotTools:

    - `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED` — set to `true`
    - `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` — set to the path of your **Label Studio Document Root**

The **Label Studio Document Root** is the folder where all datasets in Label Studio format are stored.
These variables allow AgribotTools to access and locate local files for import into Label Studio.

You can set them in a **`.env` file** placed in the root directory of AgribotTools:

```env
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/path/to/label_studio_data
```

---

## Label Studio Setup

If Label Studio is not installed, follow the [official installation guide](https://labelstud.io/guide/install.html).

---

## First Conversion

Once installed, try a conversion using one of the CLI scripts:

```bash
# Convert from YOLO segmentation to Label Studio format
python scripts/yolo_to_ls.py seg /path/to/yolo_dataset --ls_base_name my_ls_output
```

To see all available options for any script:

```bash
python scripts/<script_name>.py -h
```

For more details, see the [Conversions Usage Guide](conversions/usage.md).
