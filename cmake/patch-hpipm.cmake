# Patches applied to HPIPM before it's configured, run via `cmake -P` (as an
# ExternalProject PATCH_COMMAND, so its working directory is HPIPM's source
# directory) so the edits don't depend on shell/sed quoting.
#
# 1. The OCP QP Riccati factorization restarts with a larger primal
#    regularization whenever a pivot is singular, with no limit on the
#    number of restarts. QP data that is not finite, or that overflows during
#    the recursion, never becomes positive definite, so the solver loops
#    forever. The restarts are capped, after which the factorization carries
#    on and the solver returns with a failed status.

file(READ "ocp_qp/x_ocp_qp_kkt.c" _kkt)
if(NOT _kkt MATCHES "preg_restarts")
    string(REPLACE
        "\tint preg_first = 0;\n"
        "\tint preg_first = 0;\n\tint preg_restarts = 0;\n"
        _kkt "${_kkt}")
    string(REPLACE
        "compute_preg:\n"
        "compute_preg:\n\n\tpreg_restarts++;\n"
        _kkt "${_kkt}")
    string(REPLACE
        "goto compute_preg;"
        "if(preg_restarts < 100) goto compute_preg;"
        _kkt "${_kkt}")
    file(WRITE "ocp_qp/x_ocp_qp_kkt.c" "${_kkt}")
endif()
