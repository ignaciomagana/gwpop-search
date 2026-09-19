"""The dynesty production paths must not import NumPyro or JAXNS."""

import json
import subprocess
import sys


def test_production_search_null_and_fidelity_paths_import_no_numpyro_or_jaxns():
    code = (
        "import json, sys\n"
        "import gwpop_search.production, gwpop_search.search, gwpop_search.nulls\n"
        "import gwpop_search.inference.fidelity, gwpop_search.inference.evidence_campaign\n"
        "import gwpop_search.scouts.baseline, gwpop_search.scouts.comparison\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('numpyro', 'jaxns', 'dynesty'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    imported = json.loads(result.stdout.strip().splitlines()[-1])
    assert not [name for name in imported if name.split(".")[0] in ("numpyro", "jaxns")]
    # dynesty itself is imported lazily, only when a run starts.
    assert not [name for name in imported if name.split(".")[0] == "dynesty"]
