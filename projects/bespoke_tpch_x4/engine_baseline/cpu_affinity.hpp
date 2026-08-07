#pragma once

#include <errno.h>
#include <sched.h>
#include <stdexcept>
#include <system_error>
#include <unistd.h>



void pin_process_to_cpu(int core_id) {
    if (core_id < 0) {
        throw std::out_of_range("core_id must be non-negative");
    }

    long cpu_count = ::sysconf(_SC_NPROCESSORS_CONF);
    if (cpu_count <= 0) {
        throw std::runtime_error("Failed to query CPU count");
    }

    if (core_id >= cpu_count) {
        throw std::out_of_range("core_id exceeds available CPU count");
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);

    if (::sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        throw std::system_error(errno, std::generic_category(),
                                "sched_setaffinity failed");
    }
}

void unpin_process_from_cpus() {
    long cpu_count = ::sysconf(_SC_NPROCESSORS_CONF);
    if (cpu_count <= 0) {
        throw std::runtime_error("Failed to query CPU count");
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);

    for (int i = 0; i < cpu_count; ++i) {
        CPU_SET(i, &mask);
    }

    if (::sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        throw std::system_error(errno, std::generic_category(),
                                "sched_setaffinity reset failed");
    }
}
