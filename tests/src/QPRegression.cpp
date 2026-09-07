#include "Regression.hpp"
#include "Checks.hpp"
#include "LinearProblem.hpp"
#include <Eigen/LU>
#include <limits>

namespace ssqp
{
namespace tests
{
namespace
{
using solvers::QPData;
using solvers::QPSolver;

QPData makeQP(uint horizon, bool bounded)
{
    model::Dimensions dims {};
    dims.horizon = horizon;
    dims.stage_state.assign(horizon + 1, 2);
    dims.stage_control.assign(horizon, 2);
    dims.stage_constraint.assign(horizon + 1, bounded ? 1 : 0);
    dims.state = 2 * (horizon + 1);
    dims.control = 2 * horizon;
    dims.constraint = bounded ? horizon + 1 : 0;
    QPData qp(dims);
    for (uint k = 0; k < horizon; ++k)
    {
        auto &s = qp.stages[k];
        s.A = (MatX(2, 2) << 1, 0.1, -0.05, 0.9).finished();
        s.B = (MatX(2, 2) << 0.8, 0.1, 0.2, 0.7).finished();
        s.gamma = (VecX(2) << 0.2, -0.1).finished();
        s.Qxx = MatX::Identity(2, 2) * (1.0 + k);
        // Deliberately nonsymmetric: catches wrong cross-Hessian layout.
        s.Qxu = (MatX(2, 2) << 0.1, 0.2, -0.15, 0.05).finished();
        s.Quu = 2 * MatX::Identity(2, 2);
        s.qx = (VecX(2) << 0.1, -0.2).finished();
        s.qu = (VecX(2) << -2.0, 0.5).finished();
        s.D = MatX::Zero(bounded ? 1 : 0, 2);
        s.E = MatX::Zero(bounded ? 1 : 0, 2);
        if (bounded) { s.D << 0.1, 0; s.E << 1, 0.2; }
        s.cl = VecX::Constant(bounded ? 1 : 0, -0.25);
        s.cu = VecX::Constant(bounded ? 1 : 0, 0.25);
    }
    qp.Qxx_t = 3 * MatX::Identity(2, 2);
    qp.qx_t = (VecX(2) << -0.2, 0.3).finished();
    qp.Dt = MatX::Zero(bounded ? 1 : 0, 2);
    if (bounded) qp.Dt << 1, 0;
    qp.cl_t = VecX::Constant(bounded ? 1 : 0, -0.3);
    qp.cu_t = VecX::Constant(bounded ? 1 : 0, 0.3);
    return qp;
}

struct DenseProblem
{
    MatX H, C;
    VecX g, lower, upper;
    std::vector<MatX> X;
    std::vector<VecX> offset;
};

// Independent oracle: eliminate states, then enumerate all active sets of this
// small convex QP. No library QP, Riccati, or SQP routines are used here.
DenseProblem condense(const QPData &qp)
{
    int n = qp.dims.control, nc = qp.dims.constraint;
    DenseProblem d;
    d.H = MatX::Zero(n, n); d.g = VecX::Zero(n);
    d.C = MatX::Zero(nc, n);
    d.lower.resize(nc); d.upper.resize(nc);
    d.X.push_back(MatX::Zero(2, n)); d.offset.push_back(VecX::Zero(2));
    int row = 0;
    for (uint k = 0; k < qp.dims.horizon; ++k)
    {
        const auto &s = qp.stages[k];
        MatX U = MatX::Zero(2, n); U.block(0, 2*k, 2, 2).setIdentity();
        const auto &X = d.X.back(); const auto &b = d.offset.back();
        d.H += X.transpose()*s.Qxx*X + U.transpose()*s.Quu*U +
               X.transpose()*s.Qxu*U + U.transpose()*s.Qxu.transpose()*X;
        d.g += X.transpose()*(s.Qxx*b+s.qx) + U.transpose()*(s.Qxu.transpose()*b+s.qu);
        if (s.D.rows())
        {
            d.C.row(row) = s.D*X+s.E*U;
            d.lower[row] = s.cl[0]-(s.D*b)[0];
            d.upper[row++] = s.cu[0]-(s.D*b)[0];
        }
        VecX next = s.A*b+s.gamma;
        d.X.push_back(s.A*X+s.B*U); d.offset.push_back(next);
    }
    const auto &X = d.X.back(); const auto &b = d.offset.back();
    d.H += X.transpose()*qp.Qxx_t*X;
    d.g += X.transpose()*(qp.Qxx_t*b+qp.qx_t);
    if (qp.Dt.rows())
    {
        d.C.row(row) = qp.Dt*X;
        d.lower[row] = qp.cl_t[0]-(qp.Dt*b)[0];
        d.upper[row] = qp.cu_t[0]-(qp.Dt*b)[0];
    }
    return d;
}

VecX referenceSolution(const DenseProblem &d)
{
    int n=d.g.size(), nc=d.lower.size(), combinations=1;
    for (int i=0; i<nc; ++i) combinations*=3;
    double best=std::numeric_limits<double>::infinity();
    VecX solution;
    for (int code=0; code<combinations; ++code)
    {
        std::vector<int> active; std::vector<double> bounds;
        int value=code; bool invalid=false;
        for (int i=0; i<nc; ++i, value/=3)
        {
            int side=value%3;
            if (!side) continue;
            double bound=side==1?d.lower[i]:d.upper[i];
            if (!std::isfinite(bound)) { invalid=true; break; }
            active.push_back(i); bounds.push_back(bound);
        }
        if (invalid) continue;
        int na=active.size();
        MatX K=MatX::Zero(n+na,n+na); VecX rhs=VecX::Zero(n+na);
        K.topLeftCorner(n,n)=d.H; rhs.head(n)=-d.g;
        for(int i=0;i<na;++i)
        {
            K.block(n+i,0,1,n)=d.C.row(active[i]);
            K.block(0,n+i,n,1)=d.C.row(active[i]).transpose(); rhs[n+i]=bounds[i];
        }
        Eigen::FullPivLU<MatX> lu(K);
        if (!lu.isInvertible()) continue;
        VecX candidate=lu.solve(rhs).head(n);
        VecX c=d.C*candidate;
        if (!candidate.allFinite() || (c.array()<d.lower.array()-1e-8).any() ||
            (c.array()>d.upper.array()+1e-8).any()) continue;
        double cost=0.5*candidate.dot(d.H*candidate)+d.g.dot(candidate);
        if(cost<best) { best=cost; solution=candidate; }
    }
    require(solution.size()==n,"Dense oracle found no feasible optimum");
    return solution;
}

std::unique_ptr<QPSolver> makeSolver(const QPData &qp, solvers::ssqp::QPSolverType backend, bool limited=false)
{
    auto params=accurateParams(backend);
    if (backend==solvers::ssqp::ADMM)
    {
        params.admm_params.max_iter=limited?1:10000;
        return std::make_unique<solvers::admm_qp>(qp,params.admm_params);
    }
#ifdef WITH_HPIPM
    params.hpipm_params.max_iter=limited?1:100;
    return std::make_unique<solvers::hpipm_qp>(qp,params.hpipm_params);
#else
    throw std::runtime_error("HPIPM disabled");
#endif
}

void checkQP(const QPData &qp, const QPSolver::Solution &sol)
{
    auto d=condense(qp); VecX expected=referenceSolution(d);
    requireClose(sol.du.vec,expected,"QP vs dense active-set optimum",2e-5,2e-5);
    require(sol.lambda.vec.allFinite() && sol.mu.vec.allFinite(),"QP finite multipliers");
    for(uint k=0;k<=qp.dims.horizon;++k)
        requireClose(sol.dx.stage(k),d.X[k]*expected+d.offset[k],"QP state vs dense oracle",2e-5,2e-5);
    VecX c=d.C*sol.du.vec;
    require((c.array()>=d.lower.array()-2e-6).all() && (c.array()<=d.upper.array()+2e-6).all(),"QP bounds feasible");
    requireClose(d.H*sol.du.vec+d.g-d.C.transpose()*sol.mu.vec,VecX::Zero(d.g.size()),"Reduced KKT stationarity",3e-5,0);
    for (int i=0;i<c.size();++i)
    {
        const double mu=sol.mu.vec[i];
        double slack=mu>=0?c[i]-d.lower[i]:d.upper[i]-c[i];
        if(std::isfinite(slack)) require(std::abs(mu*slack)<3e-5,"QP complementarity");
        else require(std::abs(mu)<3e-5,"Infinite bound has zero multiplier");
    }
    for(uint k=0;k<qp.dims.horizon;++k)
    {
        const auto &s=qp.stages[k];
        requireClose(sol.dx.stage(k+1),s.A*sol.dx.stage(k)+s.B*sol.du.stage(k)+s.gamma,"QP dynamics",2e-6,0);
        requireClose(s.Quu*sol.du.stage(k)+s.qu+s.Qxu.transpose()*sol.dx.stage(k)+s.B.transpose()*sol.lambda.stage(k)-s.E.transpose()*sol.mu.stage(k),VecX::Zero(2),"Control KKT stationarity",3e-5,0);
    }
}
void qpRegression(solvers::ssqp::QPSolverType backend, uint horizon, bool bounded)
{
    auto qp=makeQP(horizon,bounded);
    auto solver=makeSolver(qp,backend);
    QPSolver::Solution sol(qp.dims);
    for(int iteration=0;iteration<5;++iteration)
    {
        // The same object must consume changed matrices, costs, affine dynamics,
        // and infinite-bound masks on reset, without stale numeric QP data.
        for(auto &s:qp.stages)
        {
            s.qu *= -0.8; s.gamma *= -0.7; s.Quu.diagonal().array() += 0.1;
            if(bounded)
            {
                s.cl[0]=iteration%2?-std::numeric_limits<double>::infinity():-0.25;
                s.cu[0]=iteration%2?0.25:std::numeric_limits<double>::infinity();
            }
        }
        solver->reset();
        require(solver->solve(sol)==solvers::SUCCESS,"QP must report success");
        checkQP(qp,sol);
    }
}
}
void addQPRegressionTests(TestContainer &tests)
{
    std::vector<solvers::ssqp::QPSolverType> backends {solvers::ssqp::ADMM};
#ifdef WITH_HPIPM
    backends.push_back(solvers::ssqp::HPIPM);
#endif
    for(auto backend:backends)
    {
        std::string suffix=backend==solvers::ssqp::ADMM?"ADMM":"HPIPM";
        for(uint horizon : {1u,3u}) for(bool bounded : {false,true})
            tests.addTest(std::make_shared<FunctionTest>("Regression_QP_Dense_H"+std::to_string(horizon)+(bounded?"_Bounded_":"_Free_")+suffix,[=]{qpRegression(backend,horizon,bounded);}));
        tests.addTest(std::make_shared<FunctionTest>("Regression_QP_IterationLimit_"+suffix,[=]{
            auto qp=makeQP(3,true); auto solver=makeSolver(qp,backend,true);
            QPSolver::Solution sol(qp.dims); solver->reset();
            require(solver->solve(sol)==solvers::MAX_ITER_REACHED,"Iteration limit must not report success");
        }));
        tests.addTest(std::make_shared<FunctionTest>("Regression_QP_Lifetime_"+suffix,[=]{
            for(int i=0;i<20;++i)
            {
                auto qp=makeQP(1+i%3,true); auto solver=makeSolver(qp,backend);
                QPSolver::Solution sol(qp.dims);
                // Both direct first solve and explicit reset are supported.
                if(i%2) solver->reset();
                require(solver->solve(sol)==solvers::SUCCESS,"First solve after construction");
                checkQP(qp,sol);
            }
        }));
    }
}
}
}
