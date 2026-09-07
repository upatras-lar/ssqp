#ifndef SSQP_TESTS_TEST_HPP
#define SSQP_TESTS_TEST_HPP

#include <random>
#include <sstream>
#include <filesystem>
#include <functional>

#include "types.hpp"

namespace ssqp
{
namespace tests
{
class Test
{
  public:
    Test(const std::string &name);
    virtual ~Test() = default;

    virtual bool run() = 0;

    const std::vector<std::string> what() const;

    const std::string name() const;

    void seed(uint32_t seed);

  protected:
    const std::string        _name;
    std::vector<std::string> _failed_logs;
    std::mt19937             _rng { 42 };
    VecX                     randomVector(uint size, double scale = 1.0);
};

class TestContainer
{
  public:
    void addTest(const std::shared_ptr<Test> &test);

    bool runAll(const std::string &filter = "", uint32_t seed = 42);
    void list() const;

  protected:
    std::vector<std::shared_ptr<Test>> _tests;
};

// A small adapter for correctness tests whose failed checks throw with context.
class FunctionTest : public Test
{
  public:
    FunctionTest(const std::string &name, std::function<void()> function);
    bool run() override;

  private:
    std::function<void()> _function;
};
}
}

#endif
