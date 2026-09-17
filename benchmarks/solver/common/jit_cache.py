"""Shared --jit ca.Function options for CasadiProblem/CasadiDuffingProblem.

CasADi's default JIT temp filename is randomized, and that randomness
leaks into the generated C source itself, so no two compiles are ever
byte-identical -- defeating a C compiler cache across a --jit run of the
same problem at several tolerances. This pins the filename instead.
"""

import hashlib
import os


def base_jit_options(opt_level="O3"):
    """Base "jit": True ca.Function options, before jit_function_options
    adds the per-instance cache directory.

    gcc's optimization passes scale worse than linearly with the size of
    CasADi's generated (flat, unrolled) C code, so a large enough problem
    needs a lower opt_level (--jit-opt-level) to compile at all.
    """
    return {
        "jit": True,
        "compiler": "shell",
        "jit_options": {"compiler": "ccache gcc", "flags": [f"-{opt_level}"]},
    }


def jit_function_options(base_options, jit_requested, *identity_parts):
    """Per-instance ca.Function options for a --jit-enabled problem.

    Pins the JIT temp filename (jit_temp_suffix=False) so repeated
    compiles of the same problem (same identity_parts, e.g. urdf path/
    horizon/free_flyer) hit ccache instead of recompiling. Each identity
    gets its own directory, so differently-identified problems running
    concurrently (--jobs > 1) never race on the same temp file.
    """
    if not jit_requested:
        return {}

    tag = hashlib.sha1(
        "|".join(str(part) for part in identity_parts).encode()
    ).hexdigest()[:16]
    directory = os.path.join(".jit_cache", tag) + os.sep
    os.makedirs(directory, exist_ok=True)

    options = dict(base_options)
    options["jit_temp_suffix"] = False
    # Keep compiled artifacts around for ccache to find next run, instead
    # of FunctionInternal's own destructor cleanup deleting them (which
    # also warns "Failed to remove" once one of several ca.Function
    # objects sharing a file -- see named_jit_options -- beats it to it)
    options["jit_cleanup"] = False
    options["jit_options"] = dict(base_options["jit_options"])
    options["jit_options"]["directory"] = directory
    return options


def named_jit_options(jit_opts, name):
    """jit_opts (see jit_function_options), pinned to this ca.Function's
    own name, so its compiled artifact doesn't share a file with every
    other cached ca.Function in the same problem.
    """
    if not jit_opts:
        return jit_opts
    return {**jit_opts, "jit_name": name}
