#include "tests/include/ADMMTest.hpp"

namespace ssqp
{
namespace tests
{
ADMMTest::ADMMTest(const std::string &name, const std::string &data_file,
                   const double tolerance)
  : ADMMTest(name, ADMMTest::_loadQPDataFromFile(data_file), tolerance)
{
}

ADMMTest::ADMMTest(
  const std::string                                            &name,
  const std::pair<solvers::QPData, solvers::admm_qp::Solution> &pair,
  const double                                                  tolerance)
  : Test(name)
  , _qp(pair.first)
  , _solution(pair.second)
  , _tol(tolerance)
{
}

bool ADMMTest::run()
{
    solvers::admm_qp           qp_solver(_qp);
    solvers::admm_qp::Solution sol(_qp.dims);
    qp_solver.solve(sol);

    if ((sol.dx - _solution.dx).vec.lpNorm<Eigen::Infinity>() > _tol)
    {
        _solution.dx.setPrintColumns(-1);
        sol.dx.setPrintColumns(-1);

        std::stringstream ss;
        ss << "Solution dx mismatch" << std::endl;
        ss << "Expected dx: " << std::endl << _solution.dx << std::endl;
        ss << "Calculated dx: " << std::endl << sol.dx << std::endl;

        _failed_logs.push_back(ss.str());
        return false;
    }

    // Note: Comparing u, z, y is pointless, since they can be different

    return true;
}

std::pair<solvers::QPData, solvers::admm_qp::Solution>
ADMMTest::_loadQPDataFromFile(const std::string &data_file)
{
    using namespace nlohmann;

    std::ifstream f(data_file);
    json          data = json::parse(f);

    model::Dimensions dims;
    dims.horizon = data["properties"]["horizon"];

    solvers::QPData qp(dims);

    const auto get_vector = [data](json v_data, VecX &v)
    {
        uint rows = v_data["rows"];
        v         = VecX(rows);

        for (uint row = 0; row < rows; ++row)
        {
            switch (v_data["data"][row][0].type())
            {
            case json::value_t::number_float:
                v[row] = v_data["data"][row][0];
                break;
            case json::value_t::string:
            default:
                v[row] = std::numeric_limits<double>::infinity();
                if (v_data["data"][row][0] == "-inf")
                {
                    v[row] *= -1;
                }
            }
        }
    };

    const auto get_matrix = [data](json M_data, MatX &M)
    {
        uint rows = M_data["rows"];
        uint cols = M_data["cols"];
        M         = MatX(rows, cols);

        for (uint row = 0; row < rows; ++row)
        {
            for (uint col = 0; col < cols; ++col)
            {
                M(row, col) = M_data["data"][row][col];
                switch (M_data["data"][row][col].type())
                {
                case json::value_t::number_float:
                    M(row, col) = M_data["data"][row][col];
                    break;
                case json::value_t::string:
                default:
                    M(row, col) = std::numeric_limits<double>::infinity();
                    if (M_data["data"][row][col] == "-inf")
                    {
                        M(row, col) *= -1;
                    }
                }
            }
        }
    };

    qp.dims.state      = 0;
    qp.dims.control    = 0;
    qp.dims.constraint = 0;

    qp.dims.stage_state.resize(dims.horizon + 1);
    qp.dims.stage_control.resize(dims.horizon);
    qp.dims.stage_constraint.resize(dims.horizon + 1);

    for (uint k = 0; k < dims.horizon; ++k)
    {
        get_matrix(data["stage_data"][k]["A"], qp.stages[k].A);
        get_matrix(data["stage_data"][k]["B"], qp.stages[k].B);
        get_vector(data["stage_data"][k]["gamma"], qp.stages[k].gamma);
        get_matrix(data["stage_data"][k]["Qxx"], qp.stages[k].Qxx);
        get_matrix(data["stage_data"][k]["Qxu"], qp.stages[k].Qxu);
        get_matrix(data["stage_data"][k]["Quu"], qp.stages[k].Quu);
        get_vector(data["stage_data"][k]["qx"], qp.stages[k].qx);
        get_vector(data["stage_data"][k]["qu"], qp.stages[k].qu);

        if (data["stage_data"][k].contains("D"))
        {
            get_matrix(data["stage_data"][k]["D"], qp.stages[k].D);
            get_matrix(data["stage_data"][k]["E"], qp.stages[k].E);
            get_vector(data["stage_data"][k]["cl"], qp.stages[k].cl);
            get_vector(data["stage_data"][k]["cu"], qp.stages[k].cu);
        }

        qp.dims.stage_state[k]      = qp.stages[k].A.rows();
        qp.dims.stage_control[k]    = qp.stages[k].B.cols();
        qp.dims.stage_constraint[k] = qp.stages[k].D.rows();

        qp.dims.state      += qp.dims.stage_state[k];
        qp.dims.control    += qp.dims.stage_control[k];
        qp.dims.constraint += qp.dims.stage_constraint[k];
    }

    get_matrix(data["terminal_data"]["Qxx"], qp.Qxx_t);
    get_vector(data["terminal_data"]["qx"], qp.qx_t);

    if (data["terminal_data"].contains("D"))
    {
        get_matrix(data["terminal_data"]["D"], qp.Dt);
        get_vector(data["terminal_data"]["cl"], qp.cl_t);
        get_vector(data["terminal_data"]["cu"], qp.cu_t);
    }

    qp.dims.stage_state[qp.dims.horizon]      = qp.Qxx_t.rows();
    qp.dims.stage_constraint[qp.dims.horizon] = qp.Dt.rows();

    qp.dims.state      += qp.dims.stage_state[qp.dims.horizon];
    qp.dims.constraint += qp.dims.stage_constraint[qp.dims.horizon];

    solvers::admm_qp::Solution sol(qp.dims);

    for (uint k = 0; k < dims.horizon; ++k)
    {
        VecX dx, du, z, y;
        get_vector(data["stage_data"][k]["dx"], dx);
        get_vector(data["stage_data"][k]["du"], du);
        get_vector(data["stage_data"][k]["z"], z);
        get_vector(data["stage_data"][k]["y"], y);

        sol.dx.stage(k) = dx;
        sol.du.stage(k) = du;
        sol.z.stage(k)  = z;
        sol.y.stage(k)  = y;
    }

    VecX dx, z, y;
    get_vector(data["terminal_data"]["dx"], dx);
    get_vector(data["terminal_data"]["z"], z);
    get_vector(data["terminal_data"]["y"], y);

    sol.dx.stage(qp.dims.horizon) = dx;
    sol.z.stage(qp.dims.horizon)  = z;
    sol.y.stage(qp.dims.horizon)  = y;

    return { qp, sol };
}
}
}