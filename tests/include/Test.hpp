#ifndef SSQP_TESTS_TEST_HPP
#define SSQP_TESTS_TEST_HPP

#include <random>
#include <sstream>

#include "types.hpp"

namespace ssqp
{
namespace tests
{
class Test
{
  public:
    Test(const std::string &name);

    virtual bool run() = 0;

    const std::vector<std::string> what() const;

    const std::string name() const;

  protected:
    const std::string        _name;
    std::vector<std::string> _failed_logs;
};

class TestContainer
{
  public:
    void addTest(const std::shared_ptr<Test> &test);

    void runAll();

  protected:
    std::vector<std::shared_ptr<Test>> _tests;
};
}
}

#endif