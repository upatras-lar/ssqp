#include "Test.hpp"

#include <fstream>
#include <iomanip>
#include <unordered_set>

namespace ssqp
{
namespace tests
{
namespace
{
std::filesystem::path failureLogPath(const std::string &test_name)
{
    return std::filesystem::path("tests/logs") / (test_name + ".txt");
}

void writeFailureLog(const std::string              &test_name,
                     const std::vector<std::string> &logs)
{
    const std::filesystem::path path = failureLogPath(test_name);
    std::error_code              error;
    std::filesystem::create_directories(path.parent_path(), error);

    if (error)
    {
        std::cout << "Failed to create log directory "
                  << path.parent_path().string() << ": " << error.message()
                  << '\n';
        return;
    }

    std::ofstream output(path);
    if (!output.is_open())
    {
        std::cout << "Failed to create file " << path.string() << '\n';
        return;
    }

    for (const auto &log : logs) output << log;

    std::cout << "Failed test " << test_name << ": Logs written to "
              << path.string() << '\n';
}

void removeFailureLog(const std::string &test_name)
{
    std::error_code error;
    std::filesystem::remove(failureLogPath(test_name), error);
}
}

Test::Test(const std::string &name) : _name(name) {}
const std::vector<std::string> Test::what() const
{
    return _failed_logs;
}
const std::string Test::name() const
{
    return _name;
}
void Test::seed(uint32_t value)
{
    _rng.seed(value);
    _failed_logs.clear();
}
VecX Test::randomVector(uint size, double scale)
{
    VecX result(size);
    // Explicit mapping avoids implementation-dependent uniform distributions.
    for (uint i = 0; i < size; ++i)
        result[i] = scale * (2.0 * (_rng() / 4294967296.0) - 1.0);
    return result;
}
void TestContainer::addTest(const std::shared_ptr<Test> &test)
{
    _tests.push_back(test);
}
void TestContainer::list() const
{
    std::unordered_set<std::string> seen;
    for (const auto &test : _tests)
        if (seen.insert(test->name()).second) std::cout << test->name() << '\n';
}
bool TestContainer::runAll(const std::string &filter, uint32_t base_seed)
{
    uint passed = 0, selected = 0;
    std::cout << "Base seed: " << base_seed << "\n";
    for (const auto &test : _tests)
    {
        if (!filter.empty() && test->name().find(filter) == std::string::npos)
            continue;
        ++selected;
        // Stable per-test seeds: running a case alone reproduces a full-suite run.
        uint32_t seed = 2166136261u;
        for (unsigned char ch : test->name()) seed = (seed ^ ch) * 16777619u;
        seed ^= base_seed;
        test->seed(seed);
        std::cout << test->name() << " (seed " << seed << "): " << std::flush;
        bool ok = false;
        std::vector<std::string> logs;
        try
        {
            ok = test->run();
        }
        catch (const std::exception &e)
        {
            logs.push_back("Exception: ");
            logs.push_back(e.what());
            logs.push_back("\n");
        }
        catch (...)
        {
            logs.push_back("Unknown exception\n");
        }
        std::cout << (ok ? "PASSED" : "FAILED") << '\n';
        if (ok)
        {
            ++passed;
            removeFailureLog(test->name());
        }
        else
        {
            const auto test_logs = test->what();
            logs.insert(logs.end(), test_logs.begin(), test_logs.end());
            if (logs.empty()) logs.push_back("Test returned false\n");
            writeFailureLog(test->name(), logs);
        }
    }
    std::cout << "Passed: " << passed << '/' << selected << '\n';
    if (!selected) std::cout << "No tests matched: " << filter << '\n';
    return selected != 0 && passed == selected;
}
FunctionTest::FunctionTest(const std::string    &name,
                           std::function<void()> function)
  : Test(name)
  , _function(std::move(function))
{
}
bool FunctionTest::run()
{
    _failed_logs.clear();
    try
    {
        _function();
        return true;
    }
    catch (const std::exception &e)
    {
        _failed_logs.push_back(e.what());
        return false;
    }
}
}
}
