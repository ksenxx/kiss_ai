#include "query_impl.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "args_parser.hpp"
#include "audit.hpp"
#include "cpu_affinity.hpp"
#include "trace_utils.hpp"
#include "query_q1.hpp"
#include "query_q1.cpp"
#include "query_q2.hpp"
#include "query_q2.cpp"
#include "query_q3.hpp"
#include "query_q3.cpp"
#include "query_q4.hpp"
#include "query_q4.cpp"
#include "query_q5.hpp"
#include "query_q5.cpp"
#include "query_q6.hpp"
#include "query_q6.cpp"
#include "query_q7.hpp"
#include "query_q7.cpp"
#include "query_q8.hpp"
#include "query_q8.cpp"
#include "query_q9.hpp"
#include "query_q9.cpp"
#include "query_q10.hpp"
#include "query_q10.cpp"
#include "query_q11.hpp"
#include "query_q11.cpp"
#include "query_q12.hpp"
#include "query_q12.cpp"
#include "query_q13.hpp"
#include "query_q13.cpp"
#include "query_q14.hpp"
#include "query_q14.cpp"
#include "query_q15.hpp"
#include "query_q15.cpp"
#include "query_q16.hpp"
#include "query_q16.cpp"
#include "query_q17.hpp"
#include "query_q17.cpp"
#include "query_q18.hpp"
#include "query_q18.cpp"
#include "query_q19.hpp"
#include "query_q19.cpp"
#include "query_q20.hpp"
#include "query_q20.cpp"
#include "query_q21.hpp"
#include "query_q21.cpp"
#include "query_q22.hpp"
#include "query_q22.cpp"


namespace {
class AffinityGuard {
public:
    explicit AffinityGuard(int cpu_id) { pin_process_to_cpu(cpu_id); }
    ~AffinityGuard() { unpin_process_from_cpus(); }

    AffinityGuard(const AffinityGuard&) = delete;
    AffinityGuard& operator=(const AffinityGuard&) = delete;
};

#ifdef TRACE
template <typename Func>
auto run_with_trace_output(Func&& func) -> decltype(func()) {
    trace_utils::TraceOutputGuard guard;
    return func();
}
#else
template <typename Func>
auto run_with_trace_output(Func&& func) -> decltype(func()) {
    return func();
}
#endif

std::string escape_csv_field(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (char ch : value) {
        if (ch == '"') {
            out.push_back('\\');
            out.push_back('"');
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('"');
    return out;
}

void write_stub_csv(size_t run_index, const QueryRequest& request) {
    const std::string filename = "result" + std::to_string(run_index) + ".csv";
    std::ofstream out(filename);
    out << "query_id,params\n";
    out << escape_csv_field(request.id) << "," << escape_csv_field(request.line) << "\n";
}

void parse_request_stub(const QueryRequest& request) {
    if (request.id == "7") {
        (void)parse_q7(request);
    } else if (request.id == "8") {
        (void)parse_q8(request);
    } else if (request.id == "9") {
        (void)parse_q9(request);
    } else if (request.id == "10") {
        (void)parse_q10(request);
    } else if (request.id == "11") {
        (void)parse_q11(request);
    } else if (request.id == "12") {
        (void)parse_q12(request);
    } else if (request.id == "13") {
        (void)parse_q13(request);
    } else if (request.id == "14") {
        (void)parse_q14(request);
    } else if (request.id == "15") {
        (void)parse_q15(request);
    } else if (request.id == "16") {
        (void)parse_q16(request);
    } else if (request.id == "17") {
        (void)parse_q17(request);
    } else if (request.id == "18") {
        (void)parse_q18(request);
    } else if (request.id == "19") {
        (void)parse_q19(request);
    } else if (request.id == "20") {
        (void)parse_q20(request);
    } else if (request.id == "21") {
        (void)parse_q21(request);
    } else if (request.id == "22") {
        (void)parse_q22(request);
    }
}
}  // namespace

void query(Database* db) {
    std::vector<QueryRequest> requests;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            break;
        }
        std::istringstream iss(line);
        std::string query_id = "0";
        iss >> query_id;
        if (!iss) {
            continue;
        }
        requests.push_back(QueryRequest{query_id, line});
    }

    for (size_t idx = 0; idx < requests.size(); ++idx) {
        const auto& request = requests[idx];
#ifdef BESPOKE_AUDIT
        std::fprintf(stderr, "AUDIT BEGIN %zu q%s\n", idx + 1, request.id.c_str());
#endif
        const auto start = std::chrono::steady_clock::now();
        if (request.id == "1") {
            Q1Args args = parse_q1(request);
            auto rows = run_with_trace_output([&]() { return run_q1(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q1_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "2") {
            Q2Args args = parse_q2(request);
            auto rows = run_with_trace_output([&]() { return run_q2(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q2_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "3") {
            Q3Args args = parse_q3(request);
            auto rows = run_with_trace_output([&]() { return run_q3(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q3_csv(*db, "result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "4") {
            Q4Args args = parse_q4(request);
            auto rows = run_with_trace_output([&]() { return run_q4(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q4_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "5") {
            Q5Args args = parse_q5(request);
            auto rows = run_with_trace_output([&]() { return run_q5(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q5_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "6") {
            Q6Args args = parse_q6(request);
            auto rows = run_with_trace_output([&]() { return run_q6(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q6_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "7") {
            Q7Args args = parse_q7(request);
            auto rows = run_with_trace_output([&]() { return run_q7(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q7_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "8") {
            Q8Args args = parse_q8(request);
            auto rows = run_with_trace_output([&]() { return run_q8(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q8_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "9") {
            Q9Args args = parse_q9(request);
            auto rows = run_with_trace_output([&]() { return run_q9(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q9_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "10") {
            Q10Args args = parse_q10(request);
            auto rows = run_with_trace_output([&]() { return run_q10(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q10_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "11") {
            Q11Args args = parse_q11(request);
            auto rows = run_with_trace_output([&]() { return run_q11(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q11_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "12") {
            Q12Args args = parse_q12(request);
            auto rows = run_with_trace_output([&]() { return run_q12(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q12_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "13") {
            Q13Args args = parse_q13(request);
            auto rows = run_with_trace_output([&]() { return run_q13(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q13_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "14") {
            Q14Args args = parse_q14(request);
            auto rows = run_with_trace_output([&]() { return run_q14(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q14_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "15") {
            Q15Args args = parse_q15(request);
            auto rows = run_with_trace_output([&]() { return run_q15(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q15_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "16") {
            Q16Args args = parse_q16(request);
            auto rows = run_with_trace_output([&]() { return run_q16(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q16_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "17") {
            Q17Args args = parse_q17(request);
            auto rows = run_with_trace_output([&]() { return run_q17(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q17_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "18") {
            Q18Args args = parse_q18(request);
            auto rows = run_with_trace_output([&]() { return run_q18(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q18_csv(*db, "result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "19") {
            Q19Args args = parse_q19(request);
            auto rows = run_with_trace_output([&]() { return run_q19(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q19_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "20") {
            Q20Args args = parse_q20(request);
            auto rows = run_with_trace_output([&]() { return run_q20(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q20_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "21") {
            Q21Args args = parse_q21(request);
            auto rows = run_with_trace_output([&]() { return run_q21(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q21_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        if (request.id == "22") {
            Q22Args args = parse_q22(request);
            auto rows = run_with_trace_output([&]() { return run_q22(*db, args); });
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration<double, std::milli>(end - start).count();
            write_q22_csv("result" + std::to_string(idx + 1) + ".csv", rows);
            std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
            continue;
        }
        parse_request_stub(request);
        const auto end = std::chrono::steady_clock::now();
        const auto elapsed =
            std::chrono::duration<double, std::milli>(end - start).count();
        write_stub_csv(idx + 1, request);
        std::cout << (idx + 1) << " | Execution ms: " << elapsed << "\n";
    }
}

// Example code for how to use the parse functions together:
//for (const auto& req : requests) {
//    switch (req.id) {
//        case "1": {
//            Q1Args args = parse_q1(req); 
//            run_q1(db, args);
//            break;
//        }
//        case "2": {
//            Q2Args args = parse_q2(req); 
//            run_q2(db, args);
//            break;
//        }
//        ...
//        case "22": {
//            Q22Args args = parse_q22(req); 
//            run_q22(db, args);
//            break;
//        }
//    }
//}
