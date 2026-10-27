#ifdef WITH_PIQP
#include "solvers/piqp_qp.hpp"

#include <piqp/piqp.hpp>

#include <algorithm>

namespace ssqp
{
namespace solvers
{

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

// The QP is passed to PIQP as
//   min 1/2 z' P z + c' z  s.t.  A z = b,  h_l <= G z <= h_u
// with z = (dx_0, du_0, ..., dx_{N-1}, du_{N-1}, dx_N), so that the
// multistage KKT solver can detect the stage structure. The rows of A are
// dx_0 = 0 followed by the dynamics A_k dx_k + B_k du_k - dx_{k+1} = -gamma_k
struct piqp_data
{
    piqp::SparseSolver<double> solver;

    // P holds the upper triangle only. All entries of the dense QP blocks are
    // stored, so the sparsity pattern stays fixed across updates
    SpMat P, A, G;
    VecX  c, b, h_l, h_u;

    // Offsets of the stage states and controls in z
    std::vector<int> x_off, u_off;
    // Offsets of the stage dynamics rows in A and constraint rows in G
    std::vector<int> dyn_off, con_off;
};

namespace
{
// Appends rows [row, row + rows) to the current column of M
void insert_rows(SpMat &M, const int col, const int row, const int rows)
{
    for (int i = 0; i < rows; ++i)
    {
        M.insertBack(row + i, col) = 0.0;
    }
}

// Writes the dense block B to M at (row, col), where each column of the block
// is stored contiguously in M
void set_block(SpMat &M, const int row, const int col, const MatX &B)
{
    for (int j = 0; j < B.cols(); ++j)
    {
        const int *begin = M.innerIndexPtr() + M.outerIndexPtr()[col + j];
        const int *end   = M.innerIndexPtr() + M.outerIndexPtr()[col + j + 1];
        const int  pos = std::lower_bound(begin, end, row) - M.innerIndexPtr();

        Eigen::Map<VecX>(M.valuePtr() + pos, B.rows()) = B.col(j);
    }
}

// Zeroes the entries of the given row of M in columns [col, col + cols)
void zero_row(SpMat &M, const int row, const int col, const int cols)
{
    for (int j = col; j < col + cols; ++j)
    {
        const int *begin = M.innerIndexPtr() + M.outerIndexPtr()[j];
        const int *end   = M.innerIndexPtr() + M.outerIndexPtr()[j + 1];

        M.valuePtr()[std::lower_bound(begin, end, row) - M.innerIndexPtr()] =
          0.0;
    }
}
}

piqp_qp::Params::Params(const nlohmann::ordered_json &piqpj)
{
    using namespace nlohmann;

    max_iter = piqpj.value("max_iter", max_iter);

    eps_abs = piqpj.value("eps_abs", eps_abs);
    eps_rel = piqpj.value("eps_rel", eps_rel);

    rho_init   = piqpj.value("rho_init", rho_init);
    delta_init = piqpj.value("delta_init", delta_init);

    reg_prim = piqpj.value("reg_prim", reg_prim);

    parallel    = piqpj.value("parallel", parallel);
    num_threads = piqpj.value("num_threads", num_threads);
}

void piqp_qp::Params::print() const
{
    std::cout << "PIQP Parameters" << std::endl;
    std::cout << "================" << std::endl;
    std::cout << "Maximum iterations: " << max_iter << std::endl;

    std::cout << "----------------" << std::endl;
    std::cout << "Absolute tolerance: " << eps_abs << std::endl;
    std::cout << "Relative tolerance: " << eps_rel << std::endl;

    std::cout << "----------------" << std::endl;
    std::cout << "Initial rho: " << rho_init << std::endl;
    std::cout << "Initial delta: " << delta_init << std::endl;
    std::cout << "Regularization weight: " << reg_prim << std::endl;

    std::cout << "----------------" << std::endl;
    std::cout << "Parallel: " << parallel << std::endl;

    if (parallel)
    {
        std::cout << "Number of threads: " << num_threads << std::endl;
    }

    std::cout << "---------------" << std::endl;
}

void piqp_qp::Params::write(nlohmann::ordered_json &j) const
{
    using namespace nlohmann;

    ordered_json piqpj;

    piqpj["max_iter"] = max_iter;

    piqpj["eps_abs"] = eps_abs;
    piqpj["eps_rel"] = eps_rel;

    piqpj["rho_init"]   = rho_init;
    piqpj["delta_init"] = delta_init;

    piqpj["reg_prim"] = reg_prim;

    piqpj["parallel"]    = parallel;
    piqpj["num_threads"] = num_threads;

    j["piqp"] = piqpj;
}

piqp_qp::piqp_qp(const QPData &qp, const Params &params)
  : _qp(qp)
  , _params(params)
  , _solution(qp.dims)
  , _reg(_params.reg_prim)
  , _piqp_data(std::make_shared<piqp_data>())
{
#ifndef PIQP_HAS_OPENMP
    if (_params.parallel)
    {
        throw std::runtime_error("PIQP: Parallel multistage solver requires "
                                 "building with SSQP_PIQP_OPENMP");
    }
#endif

    _piqp_data->x_off.resize(_qp.dims.horizon + 1);
    _piqp_data->u_off.resize(_qp.dims.horizon);
    _piqp_data->dyn_off.resize(_qp.dims.horizon);
    _piqp_data->con_off.resize(_qp.dims.horizon + 1);

    int n = 0, p = _qp.dims.stage_state[0], m = 0;
    int nnz_P = 0, nnz_A = _qp.dims.stage_state[0], nnz_G = 0;

    for (uint k = 0; k <= _qp.dims.horizon; ++k)
    {
        const int nx = _qp.dims.stage_state[k];
        const int nu = k < _qp.dims.horizon ? _qp.dims.stage_control[k] : 0;
        const int nc = _qp.dims.stage_constraint[k];

        _piqp_data->x_off[k]    = n;
        _piqp_data->con_off[k]  = m;
        n                      += nx + nu;
        m                      += nc;

        nnz_P += (nx + nu) * (nx + nu + 1) / 2;
        nnz_G += nc * (nx + nu);

        if (k < _qp.dims.horizon)
        {
            const int nx_next = _qp.dims.stage_state[k + 1];

            _piqp_data->u_off[k]    = _piqp_data->x_off[k] + nx;
            _piqp_data->dyn_off[k]  = p;
            p                      += nx_next;
            nnz_A                  += nx_next * (nx + nu + 1);
        }
    }

    _piqp_data->P.resize(n, n);
    _piqp_data->A.resize(p, n);
    _piqp_data->G.resize(m, n);
    _piqp_data->P.reserve(nnz_P);
    _piqp_data->A.reserve(nnz_A);
    _piqp_data->G.reserve(nnz_G);

    // Build the sparsity patterns column by column, in increasing row order
    for (uint k = 0; k <= _qp.dims.horizon; ++k)
    {
        const int nx = _qp.dims.stage_state[k];
        const int nu = k < _qp.dims.horizon ? _qp.dims.stage_control[k] : 0;

        for (int j = 0; j < nx + nu; ++j)
        {
            const int col = _piqp_data->x_off[k] + j;

            _piqp_data->P.startVec(col);
            insert_rows(_piqp_data->P, col, _piqp_data->x_off[k], j + 1);

            _piqp_data->A.startVec(col);
            if (j < nx)
            {
                // Initial state row, or the -I of the previous stage dynamics
                if (k == 0)
                {
                    _piqp_data->A.insertBack(j, col) = 1.0;
                }
                else
                {
                    _piqp_data->A.insertBack(_piqp_data->dyn_off[k - 1] + j,
                                             col) = -1.0;
                }
            }
            if (k < _qp.dims.horizon)
            {
                insert_rows(_piqp_data->A,
                            col,
                            _piqp_data->dyn_off[k],
                            _qp.dims.stage_state[k + 1]);
            }

            _piqp_data->G.startVec(col);
            insert_rows(_piqp_data->G,
                        col,
                        _piqp_data->con_off[k],
                        _qp.dims.stage_constraint[k]);
        }
    }

    _piqp_data->P.finalize();
    _piqp_data->A.finalize();
    _piqp_data->G.finalize();

    _piqp_data->c.resize(n);
    _piqp_data->b = VecX::Zero(p);
    _piqp_data->h_l.resize(m);
    _piqp_data->h_u.resize(m);

    auto &settings      = _piqp_data->solver.settings();
    settings.kkt_solver = piqp::KKTSolver::sparse_multistage;
#ifdef PIQP_HAS_OPENMP
    // The serial solver also has OpenMP regions, which stay single threaded
    settings.num_threads = 1;
    if (_params.parallel)
    {
        settings.kkt_solver  = piqp::KKTSolver::sparse_multistage_parallel;
        settings.num_threads = _params.num_threads;
    }
#endif
    settings.max_iter   = _params.max_iter;
    settings.eps_abs    = _params.eps_abs;
    settings.eps_rel    = _params.eps_rel;
    // The duality gap has separate tolerances, which are otherwise far
    // tighter than the requested ones
    settings.eps_duality_gap_abs = _params.eps_abs;
    settings.eps_duality_gap_rel = _params.eps_rel;
    settings.rho_init   = _params.rho_init;
    settings.delta_init = _params.delta_init;

    // reset() uploads numeric QP data. The QP is assembled by SSQP after this
    // constructor returns, so defer that upload until the first solve.
}

piqp_qp::~piqp_qp() {}

void piqp_qp::reset()
{
    _reg = _params.reg_prim;

    for (uint k = 0; k <= _qp.dims.horizon; ++k)
    {
        const int   nx   = _qp.dims.stage_state[k];
        const MatX &Qxx  = k < _qp.dims.horizon ? _qp.stages[k].Qxx : _qp.Qxx_t;
        double     *vals = _piqp_data->P.valuePtr();

        // Each column of P starts at the first row of its stage
        for (int i = 0; i < nx; ++i)
        {
            double *v =
              vals + _piqp_data->P.outerIndexPtr()[_piqp_data->x_off[k] + i];
            Eigen::Map<VecX>(v, i + 1)  = Qxx.col(i).head(i + 1);
            v[i]                       += _reg;
        }

        if (k == _qp.dims.horizon) break;

        const QPStage &s = _qp.stages[k];

        for (int j = 0; j < _qp.dims.stage_control[k]; ++j)
        {
            double *v =
              vals + _piqp_data->P.outerIndexPtr()[_piqp_data->u_off[k] + j];
            Eigen::Map<VecX>(v, nx)          = s.Qxu.col(j);
            Eigen::Map<VecX>(v + nx, j + 1)  = s.Quu.col(j).head(j + 1);
            v[nx + j]                       += _reg;
        }

        set_block(_piqp_data->A,
                  _piqp_data->dyn_off[k],
                  _piqp_data->x_off[k],
                  s.A);
        set_block(_piqp_data->A,
                  _piqp_data->dyn_off[k],
                  _piqp_data->u_off[k],
                  s.B);
        _piqp_data->b.segment(_piqp_data->dyn_off[k],
                              _qp.dims.stage_state[k + 1]) = -s.gamma;

        _piqp_data->c.segment(_piqp_data->x_off[k], nx) = s.qx;
        _piqp_data->c.segment(_piqp_data->u_off[k], _qp.dims.stage_control[k]) =
          s.qu;

        if (_qp.dims.stage_constraint[k] > 0)
        {
            set_block(_piqp_data->G,
                      _piqp_data->con_off[k],
                      _piqp_data->x_off[k],
                      s.D);
            set_block(_piqp_data->G,
                      _piqp_data->con_off[k],
                      _piqp_data->u_off[k],
                      s.E);
            _piqp_data->h_l.segment(_piqp_data->con_off[k],
                                    _qp.dims.stage_constraint[k]) = s.cl;
            _piqp_data->h_u.segment(_piqp_data->con_off[k],
                                    _qp.dims.stage_constraint[k]) = s.cu;
        }
    }

    _piqp_data->c.segment(_piqp_data->x_off[_qp.dims.horizon],
                          _qp.dims.stage_state[_qp.dims.horizon]) = _qp.qx_t;

    if (_qp.dims.stage_constraint[_qp.dims.horizon] > 0)
    {
        set_block(_piqp_data->G,
                  _piqp_data->con_off[_qp.dims.horizon],
                  _piqp_data->x_off[_qp.dims.horizon],
                  _qp.Dt);
        _piqp_data->h_l.segment(_piqp_data->con_off[_qp.dims.horizon],
                                _qp.dims.stage_constraint[_qp.dims.horizon]) =
          _qp.cl_t;
        _piqp_data->h_u.segment(_piqp_data->con_off[_qp.dims.horizon],
                                _qp.dims.stage_constraint[_qp.dims.horizon]) =
          _qp.cu_t;
    }

    // Rows without finite bounds become 0 in [-1, 1], as PIQP would otherwise
    // do itself while printing a warning on every setup and update
    for (uint k = 0; k <= _qp.dims.horizon; ++k)
    {
        const int cols = _qp.dims.stage_state[k] +
                         (k < _qp.dims.horizon ? _qp.dims.stage_control[k] : 0);

        for (uint c = 0; c < _qp.dims.stage_constraint[k]; ++c)
        {
            const int row = _piqp_data->con_off[k] + c;

            if (std::isinf(_piqp_data->h_l[row]) &&
                std::isinf(_piqp_data->h_u[row]))
            {
                zero_row(_piqp_data->G, row, _piqp_data->x_off[k], cols);
                _piqp_data->h_l[row] = -1.0;
                _piqp_data->h_u[row] = 1.0;
            }
        }
    }

    if (_setup_done)
    {
        _piqp_data->solver.update(_piqp_data->P,
                                  _piqp_data->c,
                                  _piqp_data->A,
                                  _piqp_data->b,
                                  _piqp_data->G,
                                  _piqp_data->h_l,
                                  _piqp_data->h_u);
    }
    else
    {
        _piqp_data->solver.setup(_piqp_data->P,
                                 _piqp_data->c,
                                 _piqp_data->A,
                                 _piqp_data->b,
                                 _piqp_data->G,
                                 _piqp_data->h_l,
                                 _piqp_data->h_u);
        _setup_done = true;
    }

    _qp_uploaded = true;
}

void piqp_qp::increaseRegularization()
{
    const double prev_reg  = _reg;
    _reg                  *= 10;

    if (_reg > _reg_max)
    {
        throw std::runtime_error("PIQP: Regularization failed");
    }

    // The last entry of each column of the upper triangle is the diagonal
    for (int j = 0; j < _piqp_data->P.cols(); ++j)
    {
        _piqp_data->P.valuePtr()[_piqp_data->P.outerIndexPtr()[j + 1] - 1] +=
          _reg - prev_reg;
    }

    if (_setup_done)
    {
        _piqp_data->solver.update(_piqp_data->P);
    }
}

SolverState piqp_qp::solve(piqp_qp::Solution &sol)
{
    if (!_qp_uploaded)
    {
        reset();
    }

    const piqp::Status status = _piqp_data->solver.solve();
    const auto        &res    = _piqp_data->solver.result();

    _iter = res.info.iter;

    if (GlobalLogMask & LogLevel::Warning && status != piqp::PIQP_SOLVED)
    {
        logln_at(LogLevel::Warning,
                 "Warning: PIQP returned status "
                   << piqp::status_to_string(status));
    }

    logln_at(LogLevel::Debug,
             "PIQP status: " << piqp::status_to_string(status));

    // PIQP's Lagrangian has y'(A z - b) + z_u'(G z - h_u) - z_l'(G z - h_l)
    for (uint k = 0; k <= _qp.dims.horizon; ++k)
    {
        const int nc = _qp.dims.stage_constraint[k];

        sol.dx.stage(k) =
          res.x.segment(_piqp_data->x_off[k], _qp.dims.stage_state[k]);
        sol.mu.stage(k) = res.z_l.segment(_piqp_data->con_off[k], nc) -
                          res.z_u.segment(_piqp_data->con_off[k], nc);

        if (k == _qp.dims.horizon) break;

        sol.du.stage(k) =
          res.x.segment(_piqp_data->u_off[k], _qp.dims.stage_control[k]);
        sol.lambda.stage(k) =
          res.y.segment(_piqp_data->dyn_off[k], _qp.dims.stage_state[k + 1]);
    }

    switch (status)
    {
    case piqp::PIQP_MAX_ITER_REACHED:
        return MAX_ITER_REACHED;
    case piqp::PIQP_PRIMAL_INFEASIBLE:
        return PRIMAL_INFEASIBLE;
    default:
        return SUCCESS;
    }
}

uint piqp_qp::iterations() const
{
    return _iter;
}

MatX piqp_qp::feedbackGain(const uint k) const
{
    throw std::runtime_error("PIQP: Feedback gain is not available");
}

}
}
#endif
