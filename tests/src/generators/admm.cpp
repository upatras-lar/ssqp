#include "generators/admm.hpp"

namespace ssqp
{
namespace tests
{
namespace generators
{
std::vector<std::shared_ptr<Test>> get_admm_problems()
{
    std::vector<std::shared_ptr<Test>> tests;

    for (const auto &entry :
         std::filesystem::directory_iterator("tests/data/admm"))
    {
        if (entry.is_regular_file())
        {
            tests.push_back(std::make_shared<ADMMTest>(
              "ADMMTest-" + static_cast<std::string>(entry.path().filename()),
              entry.path(),
              1e-1));
        }
    }

    return tests;
}

std::shared_ptr<Test> get_admm_infeasibility()
{
    return std::make_shared<ADMMInfeasibility>("ADMMInfeasibilityTest");
}
}
}
}