#include <iostream>
#include <deque>

#include <Eigen/Core>

#include "models/double_pendulum/DoublePendulum.hpp"

#define WINDOW_SIZE 50

struct DataPoint
{
    Eigen::Vector4d x_prev;
    Eigen::Vector2d u;
    Eigen::Vector4d x_pred, x_actual;
};

int main()
{
    ssqp::model::DoublePendulumDynamics::Params params;
    params.m1 = 0.5;
    params.m2 = 0.5;
    params.l1 = 0.5;
    params.l2 = 0.5;
    params.I1 = 0.01;
    params.I2 = 0.01;

    ssqp::model::DoublePendulumDynamics dynamics(params);

    const uint STATE_SIZE = dynamics.state_size();

    std::deque<DataPoint> window;

    Eigen::Vector4d x = Eigen::Vector4d::Zero();

    for (uint iter = 0; iter < 100; ++iter)
    {
        std::cout << "Current Parameters, iteration " << iter << std::endl;
        std::cout << "------------------" << std::endl;
        std::cout << "m1: " << params.m1 << std::endl;
        std::cout << "m2: " << params.m2 << std::endl;
        std::cout << "l1: " << params.l1 << std::endl;
        std::cout << "l2: " << params.l2 << std::endl;
        std::cout << "I1: " << params.I1 << std::endl;
        std::cout << "I2: " << params.I2 << std::endl;
        std::cout << "------------------" << std::endl;

        DataPoint point;

        // Get current state from system
        point.x_prev = x;
        // Get control from solver
        point.u = Eigen::Vector2d::Zero();

        // Evaluate with current model
        point.x_pred = dynamics.evaluate(point.x_prev, point.u);

        // TODO: Apply control to real system & get actual state
        x              = Eigen::Vector4d::Zero();
        point.x_actual = x;

        // Keep new datapoint
        window.push_back(point);

        if (window.size() > WINDOW_SIZE)
        {
            window.pop_front();
        }

        if (window.size() == WINDOW_SIZE)
        {
            // 1. Initialize stacked residuals (r) and Jacobian (J)
            Eigen::VectorXd r(WINDOW_SIZE * STATE_SIZE);
            // TODO: Jacobian wrt model parameters
            Eigen::MatrixXd J(WINDOW_SIZE * STATE_SIZE, 6);

            Eigen::Vector<double, 6> theta;
            theta << params.m1, params.m2, params.l1, params.l2, params.I1,
              params.I2;

            // 2. Compute residuals and Jacobian
            for (uint i = 0; i < window.size(); ++i)
            {
                const DataPoint &p       = window[i];
                uint             row_idx = i * STATE_SIZE;

                r.segment(row_idx, STATE_SIZE) = p.x_actual - p.x_pred;

                J.block<4, 6>(i * STATE_SIZE, 0) =
                  dynamics.compute_param_jacobian(p.x_prev, p.u, p.x_actual);
            }

            // 3. Solve equations
            double          lambda  = 1e-4;
            Eigen::MatrixXd H       = J.transpose() * J;
            H.diagonal()           += lambda * H.diagonal();

            Eigen::VectorXd dTheta = H.ldlt().solve(-J.transpose() * r);

            // 4. Update parameters
            // TODO: line search
            theta += dTheta;

            for (uint i = 0; i < 6; ++i)
            {
                if (theta[i] < 1e-3)
                {
                    theta[i] = 1e-3;
                }
            }

            params.m1 = theta[0];
            params.m2 = theta[1];
            params.l1 = theta[2];
            params.l2 = theta[3];
            params.I1 = theta[4];
            params.I2 = theta[5];

            dynamics.updateParameters(params);
        }
    }

    return 0;
}