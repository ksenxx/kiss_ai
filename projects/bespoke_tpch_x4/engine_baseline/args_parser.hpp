#pragma once

#include <iomanip>
#include <string>
#include <sstream>
#include <vector>

struct QueryRequest {
    std::string id;
    std::string line;
};

// Helper function to parse IN list from tuple syntax: ('val1', 'val2', ...)
std::vector<std::string> parse_in_list(std::istringstream& iss) {
    std::vector<std::string> result;

    // Read opening parenthesis
    char c;
    iss >> std::ws >> c;
    if (c != '(') {
        std::ostringstream oss;
        oss << "Expected '(' at start of IN list, but got '"
            << c << "' (int=" << static_cast<int>(static_cast<unsigned char>(c)) << ")";
        throw std::runtime_error(oss.str());
    }

    bool first = true;
    while (iss >> std::ws) {
        // Check for closing parenthesis
        if (iss.peek() == ')') {
            iss.get(); // consume ')'
            break;
        }

        // Skip comma after first element
        if (!first) {
            iss >> std::ws >> c;
            if (c != ',') {
                throw std::runtime_error("Expected ',' between IN list elements");
            }
        }
        first = false;

        std::string value;
        iss >> std::ws;
        if (iss.peek() == '\'') {
            iss.get();
            while (iss) {
                const char ch = static_cast<char>(iss.get());
                if (!iss) break;
                if (ch == '\'') {
                    if (iss.peek() == '\'') {
                        iss.get();
                        value.push_back('\'');
                        continue;
                    }
                    break;
                }
                value.push_back(ch);
            }
        } else {
            while (iss && iss.peek() != ',' && iss.peek() != ')') {
                value.push_back(static_cast<char>(iss.get()));
            }
            const auto start = value.find_first_not_of(" \t\r\n");
            const auto end = value.find_last_not_of(" \t\r\n");
            if (start == std::string::npos) {
                value.clear();
            } else {
                value = value.substr(start, end - start + 1);
            }
        }

        result.push_back(value);
    }

    return result;
}


//Q1
struct Q1Args {
    std::string DELTA;
};
inline Q1Args parse_q1(const QueryRequest& request) {
    Q1Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q1: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.DELTA))) {
		throw std::runtime_error("Q1: failed to parse DELTA");
	}

    return args;
}
    
//Q2
struct Q2Args {
    std::string SIZE;
    std::string TYPE;
    std::string REGION;
};
inline Q2Args parse_q2(const QueryRequest& request) {
    Q2Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q2: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.SIZE))) {
		throw std::runtime_error("Q2: failed to parse SIZE");
	}
	if (!(iss >> std::quoted(args.TYPE))) {
		throw std::runtime_error("Q2: failed to parse TYPE");
	}
	if (!(iss >> std::quoted(args.REGION))) {
		throw std::runtime_error("Q2: failed to parse REGION");
	}

    return args;
}
    
//Q3
struct Q3Args {
    std::string SEGMENT;
    std::string DATE;
};
inline Q3Args parse_q3(const QueryRequest& request) {
    Q3Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q3: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.SEGMENT))) {
		throw std::runtime_error("Q3: failed to parse SEGMENT");
	}
	if (!(iss >> std::quoted(args.DATE))) {
		throw std::runtime_error("Q3: failed to parse DATE");
	}

    return args;
}
    
//Q4
struct Q4Args {
    std::string DATE;
};
inline Q4Args parse_q4(const QueryRequest& request) {
    Q4Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q4: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.DATE))) {
		throw std::runtime_error("Q4: failed to parse DATE");
	}

    return args;
}
    
//Q5
struct Q5Args {
    std::string REGION;
    std::string DATE;
};
inline Q5Args parse_q5(const QueryRequest& request) {
    Q5Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q5: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.REGION))) {
		throw std::runtime_error("Q5: failed to parse REGION");
	}
	if (!(iss >> std::quoted(args.DATE))) {
		throw std::runtime_error("Q5: failed to parse DATE");
	}

    return args;
}
    
//Q6
struct Q6Args {
    std::string DATE;
    std::string DISCOUNT;
    std::string QUANTITY;
};
inline Q6Args parse_q6(const QueryRequest& request) {
    Q6Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q6: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.DATE))) {
		throw std::runtime_error("Q6: failed to parse DATE");
	}
	if (!(iss >> std::quoted(args.DISCOUNT))) {
		throw std::runtime_error("Q6: failed to parse DISCOUNT");
	}
	if (!(iss >> std::quoted(args.QUANTITY))) {
		throw std::runtime_error("Q6: failed to parse QUANTITY");
	}

    return args;
}
    
//Q7
struct Q7Args {
    std::string NATION1;
    std::string NATION2;
};
inline Q7Args parse_q7(const QueryRequest& request) {
    Q7Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q7: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.NATION1))) {
		throw std::runtime_error("Q7: failed to parse NATION1");
	}
	if (!(iss >> std::quoted(args.NATION2))) {
		throw std::runtime_error("Q7: failed to parse NATION2");
	}

    return args;
}
    
//Q8
struct Q8Args {
    std::string NATION;
    std::string REGION;
    std::string TYPE;
};
inline Q8Args parse_q8(const QueryRequest& request) {
    Q8Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q8: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.NATION))) {
		throw std::runtime_error("Q8: failed to parse NATION");
	}
	if (!(iss >> std::quoted(args.REGION))) {
		throw std::runtime_error("Q8: failed to parse REGION");
	}
	if (!(iss >> std::quoted(args.TYPE))) {
		throw std::runtime_error("Q8: failed to parse TYPE");
	}

    return args;
}
    
//Q9
struct Q9Args {
    std::string COLOR;
};
inline Q9Args parse_q9(const QueryRequest& request) {
    Q9Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q9: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.COLOR))) {
		throw std::runtime_error("Q9: failed to parse COLOR");
	}

    return args;
}
    
//Q10
struct Q10Args {
    std::string DATE;
};
inline Q10Args parse_q10(const QueryRequest& request) {
    Q10Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q10: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.DATE))) {
		throw std::runtime_error("Q10: failed to parse DATE");
	}

    return args;
}
    
//Q11
struct Q11Args {
    std::string NATION;
    std::string FRACTION;
};
inline Q11Args parse_q11(const QueryRequest& request) {
    Q11Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q11: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.NATION))) {
		throw std::runtime_error("Q11: failed to parse NATION");
	}
	if (!(iss >> std::quoted(args.FRACTION))) {
		throw std::runtime_error("Q11: failed to parse FRACTION");
	}

    return args;
}
    
//Q12
struct Q12Args {
    std::string SHIPMODE1;
    std::string SHIPMODE2;
    std::string DATE;
};
inline Q12Args parse_q12(const QueryRequest& request) {
    Q12Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q12: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.SHIPMODE1))) {
		throw std::runtime_error("Q12: failed to parse SHIPMODE1");
	}
	if (!(iss >> std::quoted(args.SHIPMODE2))) {
		throw std::runtime_error("Q12: failed to parse SHIPMODE2");
	}
	if (!(iss >> std::quoted(args.DATE))) {
		throw std::runtime_error("Q12: failed to parse DATE");
	}

    return args;
}
    
//Q13
struct Q13Args {
    std::string WORD1;
    std::string WORD2;
};
inline Q13Args parse_q13(const QueryRequest& request) {
    Q13Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q13: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.WORD1))) {
		throw std::runtime_error("Q13: failed to parse WORD1");
	}
	if (!(iss >> std::quoted(args.WORD2))) {
		throw std::runtime_error("Q13: failed to parse WORD2");
	}

    return args;
}
    
//Q14
struct Q14Args {
    std::string DATE;
};
inline Q14Args parse_q14(const QueryRequest& request) {
    Q14Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q14: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.DATE))) {
		throw std::runtime_error("Q14: failed to parse DATE");
	}

    return args;
}
    
//Q15
struct Q15Args {
    std::string DATE;
    std::string STREAM_ID;
};
inline Q15Args parse_q15(const QueryRequest& request) {
    Q15Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q15: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.DATE))) {
		throw std::runtime_error("Q15: failed to parse DATE");
	}
	if (!(iss >> std::quoted(args.STREAM_ID))) {
		throw std::runtime_error("Q15: failed to parse STREAM_ID");
	}

    return args;
}
    
//Q16
struct Q16Args {
    std::string BRAND;
    std::string TYPE;
    std::string SIZE1;
    std::string SIZE2;
    std::string SIZE3;
    std::string SIZE4;
    std::string SIZE5;
    std::string SIZE6;
    std::string SIZE7;
    std::string SIZE8;
};
inline Q16Args parse_q16(const QueryRequest& request) {
    Q16Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q16: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.BRAND))) {
		throw std::runtime_error("Q16: failed to parse BRAND");
	}
	if (!(iss >> std::quoted(args.TYPE))) {
		throw std::runtime_error("Q16: failed to parse TYPE");
	}
	if (!(iss >> std::quoted(args.SIZE1))) {
		throw std::runtime_error("Q16: failed to parse SIZE1");
	}
	if (!(iss >> std::quoted(args.SIZE2))) {
		throw std::runtime_error("Q16: failed to parse SIZE2");
	}
	if (!(iss >> std::quoted(args.SIZE3))) {
		throw std::runtime_error("Q16: failed to parse SIZE3");
	}
	if (!(iss >> std::quoted(args.SIZE4))) {
		throw std::runtime_error("Q16: failed to parse SIZE4");
	}
	if (!(iss >> std::quoted(args.SIZE5))) {
		throw std::runtime_error("Q16: failed to parse SIZE5");
	}
	if (!(iss >> std::quoted(args.SIZE6))) {
		throw std::runtime_error("Q16: failed to parse SIZE6");
	}
	if (!(iss >> std::quoted(args.SIZE7))) {
		throw std::runtime_error("Q16: failed to parse SIZE7");
	}
	if (!(iss >> std::quoted(args.SIZE8))) {
		throw std::runtime_error("Q16: failed to parse SIZE8");
	}

    return args;
}
    
//Q17
struct Q17Args {
    std::string BRAND;
    std::string CONTAINER;
};
inline Q17Args parse_q17(const QueryRequest& request) {
    Q17Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q17: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.BRAND))) {
		throw std::runtime_error("Q17: failed to parse BRAND");
	}
	if (!(iss >> std::quoted(args.CONTAINER))) {
		throw std::runtime_error("Q17: failed to parse CONTAINER");
	}

    return args;
}
    
//Q18
struct Q18Args {
    std::string QUANTITY;
};
inline Q18Args parse_q18(const QueryRequest& request) {
    Q18Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q18: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.QUANTITY))) {
		throw std::runtime_error("Q18: failed to parse QUANTITY");
	}

    return args;
}
    
//Q19
struct Q19Args {
    std::string QUANTITY1;
    std::string QUANTITY2;
    std::string QUANTITY3;
    std::string BRAND1;
    std::string BRAND2;
    std::string BRAND3;
};
inline Q19Args parse_q19(const QueryRequest& request) {
    Q19Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q19: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.QUANTITY1))) {
		throw std::runtime_error("Q19: failed to parse QUANTITY1");
	}
	if (!(iss >> std::quoted(args.QUANTITY2))) {
		throw std::runtime_error("Q19: failed to parse QUANTITY2");
	}
	if (!(iss >> std::quoted(args.QUANTITY3))) {
		throw std::runtime_error("Q19: failed to parse QUANTITY3");
	}
	if (!(iss >> std::quoted(args.BRAND1))) {
		throw std::runtime_error("Q19: failed to parse BRAND1");
	}
	if (!(iss >> std::quoted(args.BRAND2))) {
		throw std::runtime_error("Q19: failed to parse BRAND2");
	}
	if (!(iss >> std::quoted(args.BRAND3))) {
		throw std::runtime_error("Q19: failed to parse BRAND3");
	}

    return args;
}
    
//Q20
struct Q20Args {
    std::string COLOR;
    std::string DATE;
    std::string NATION;
};
inline Q20Args parse_q20(const QueryRequest& request) {
    Q20Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q20: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.COLOR))) {
		throw std::runtime_error("Q20: failed to parse COLOR");
	}
	if (!(iss >> std::quoted(args.DATE))) {
		throw std::runtime_error("Q20: failed to parse DATE");
	}
	if (!(iss >> std::quoted(args.NATION))) {
		throw std::runtime_error("Q20: failed to parse NATION");
	}

    return args;
}
    
//Q21
struct Q21Args {
    std::string NATION;
};
inline Q21Args parse_q21(const QueryRequest& request) {
    Q21Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q21: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.NATION))) {
		throw std::runtime_error("Q21: failed to parse NATION");
	}

    return args;
}
    
//Q22
struct Q22Args {
    std::string I1;
    std::string I2;
    std::string I3;
    std::string I4;
    std::string I5;
    std::string I6;
    std::string I7;
};
inline Q22Args parse_q22(const QueryRequest& request) {
    Q22Args args;
    std::istringstream iss(request.line);

    std::string qid;
    if (!(iss >> qid)) {  // consume query id
		throw std::runtime_error("Q22: failed to parse query id"); 
    }

	if (!(iss >> std::quoted(args.I1))) {
		throw std::runtime_error("Q22: failed to parse I1");
	}
	if (!(iss >> std::quoted(args.I2))) {
		throw std::runtime_error("Q22: failed to parse I2");
	}
	if (!(iss >> std::quoted(args.I3))) {
		throw std::runtime_error("Q22: failed to parse I3");
	}
	if (!(iss >> std::quoted(args.I4))) {
		throw std::runtime_error("Q22: failed to parse I4");
	}
	if (!(iss >> std::quoted(args.I5))) {
		throw std::runtime_error("Q22: failed to parse I5");
	}
	if (!(iss >> std::quoted(args.I6))) {
		throw std::runtime_error("Q22: failed to parse I6");
	}
	if (!(iss >> std::quoted(args.I7))) {
		throw std::runtime_error("Q22: failed to parse I7");
	}

    return args;
}
    