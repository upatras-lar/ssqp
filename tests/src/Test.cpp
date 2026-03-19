#include "Test.hpp"

namespace ssqp
{
namespace tests
{
Test::Test(const std::string &name) : _name(name) {}

const std::vector<std::string> Test::what() const
{
    return _failed_logs;
}

const std::string Test::name() const
{
    return _name;
}

void TestContainer::addTest(const std::shared_ptr<Test> &test)
{
    _tests.push_back(test);
}

void TestContainer::runAll()
{
    uint npassed = 0;

    for (uint i = 0; i < _tests.size(); ++i)
    {
        std::cout << _tests[i]->name() << ": " << std::flush;

        bool passed = _tests[i]->run();

        if (passed)
        {
            std::cout << "\033[32m" << "PASSED\033[39m" << std::endl;
            ++npassed;
            continue;
        }

        std::cout << "\033[31m" << "FAILED\033[39m" << std::endl;
    }

    std::cout << "Passed: \033[32m" << npassed << "\033[39m/" << _tests.size()
              << std::endl;

    for (const auto &entry : std::filesystem::directory_iterator("tests/logs"))
    {
        std::filesystem::remove_all(entry.path());
    }

    if (npassed != _tests.size())
    {
        for (uint i = 0; i < _tests.size(); ++i)
        {
            const std::vector<std::string> logs = _tests[i]->what();

            if (logs.empty())
            {
                continue;
            }

            const std::string fname =
              "tests/logs/" + _tests[i]->name() + ".txt";
            std::ofstream of(fname);

            if (!of.is_open())
            {
                std::cout << "Failed to create file " << fname << std::endl;
                continue;
            }

            for (uint i = 0; i < logs.size(); ++i)
            {
                of << logs[i];
            }

            of.close();

            std::cout << "Failed test " << _tests[i]->name()
                      << ": Logs written to \033[31m" << fname << "\033[39m"
                      << std::endl;
        }
    }
}
}
}