# ./huginn/AGENTS.md

Always use camel case when writing code for this project.

Never try to install this package, it is over 1.5 GB. Only ever work with the raw code.


The project directory has the following structure:

huginn/
├─ src/
│  ├─ docs/
│  │  ├─ api.md
│  │  └─ architecture.md
│  ├─ examples/
│  │  └─ minimalTextGen.py
│  ├─ huginn/
│  │  ├─ aiAssistant/
│  │  │  ├─ __init__.py
│  │  │  ├─ assistant.py
│  │  │  └─ systemDiagnostics.py
│  │  ├─ c/
│  │  │  ├─ __init__.py
│  │  │  └─ meson.build
│  │  ├─ config/
│  │  │  ├─ __init__.py
│  │  │  ├─ defaults.toml
│  │  │  └─ schema.json
│  │  ├─ cython/
│  │  │  ├─ __init__.py
│  │  │  └─ setupCython.py
│  │  ├─ integration/
│  │  │  ├─ __init__.py
│  │  │  ├─ grendelClient.py
│  │  │  └─ modelRegistry.py
│  │  ├─ ipc/
│  │  │  ├─ __init__.py
│  │  │  └─ messages.py
│  │  ├─ observability/
│  │  │  ├─ __init__.py
│  │  │  ├─ logging.py
│  │  │  └─ metrics.py
│  │  ├─ orchestrator/
│  │  │  ├─ __init__.py
│  │  │  ├─ executionEngine.py
│  │  │  └─ jobManager.py
│  │  ├─ planner/
│  │  │  ├─ __init__.py
│  │  │  ├─ costModel.py
│  │  │  ├─ partitioner.py
│  │  │  └─ planGraph.py
│  │  ├─ quantization/
│  │  │  ├─ __init__.py
│  │  │  └─ loadQuantizer.py
│  │  ├─ runtime/
│  │  │  ├─ __init__.py
│  │  │  ├─ onnxAdapter.py
│  │  │  └─ torchAdapter.py
│  │  ├─ scheduling/
│  │  │  ├─ __init__.py
│  │  │  └─ policy.py
│  │  ├─ sdk/
│  │  │  ├─ __init__.py
│  │  │  └─ client.py
│  │  ├─ weights/
│  │  │  ├─ __init__.py
│  │  │  ├─ cache.py
│  │  │  └─ downloader.py
│  │  ├─ __init__.py
│  │  ├─ __main__.py
│  │  ├─ cli.py
│  │  └─ dataTypes.py
│  └─ scripts/
│     ├─ dev.ps1
│     └─ dev.sh
├─ AGENTS.md
├─ CHANGELOG.md
├─ huginn
├─ LICENSE
├─ pyproject.toml
├─ README.md
└─ setup.py