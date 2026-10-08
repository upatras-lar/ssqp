#ifndef SSQP_PYTHON_MODEL_MODEL_REFERENCE_HPP
#define SSQP_PYTHON_MODEL_MODEL_REFERENCE_HPP

#include <memory>
#include <unordered_set>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

// Call policy for the constructors of objects that refer to the model they are
// built with, like costs, constraints and solvers. The model is kept in the
// instance dictionary, so it stays alive as long as they do, while Python's
// garbage collector can still free them together with the model
struct keep_model
{
    static void precall(PyObject **, size_t, nb::detail::cleanup_list *) {}

    static void postcall(PyObject **args, size_t, nb::handle)
    {
        nb::setattr(args[0], "_model", args[1]);
    }
};

// Visits the Python object a shared pointer holds a reference to, for Python's
// garbage collector. nanobind gives C++ a Python object as a shared pointer
// whose deleter holds a reference to it, one per shared pointer it creates, so
// each deleter is visited once. Returns what Py_VISIT does
template<typename T>
int visit_python_reference(const std::shared_ptr<T>         &ptr,
                           std::unordered_set<const void *> &visited,
                           visitproc visit, void *arg)
{
    const nb::detail::py_deleter *deleter =
      std::get_deleter<nb::detail::py_deleter>(ptr);
    if (deleter != nullptr && visited.insert(deleter).second)
    {
        Py_VISIT(deleter->o);
    }

    return 0;
}
}
}

#endif
