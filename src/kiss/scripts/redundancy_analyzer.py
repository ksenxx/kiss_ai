import xml.etree.ElementTree as ET

def analyze_redundancy(coverage_file="coverage.xml"):
    tree = ET.parse(coverage_file)
    root = tree.getroot()

    coverage_data = {}
    for class_element in root.findall(".//class"):
        class_name = class_element.get("name")
        for method_element in class_element.findall(".//method"):
            method_name = method_element.get("name")
            method_signature = f"{class_name}.{method_name}"
            covered_lines = set()
            for line_element in method_element.findall(".//line"):
                if line_element.get("hits") != "0":
                    covered_lines.add(int(line_element.get("number")))
            coverage_data[method_signature] = covered_lines

    redundant_tests = []
    for test1, coverage1 in coverage_data.items():
        for test2, coverage2 in coverage_data.items():
            if test1 != test2 and coverage1.issubset(coverage2):
                redundant_tests.append((test1, test2))
                print(f"Test {test1} is redundant because it's covered by {test2}")

    if not redundant_tests:
        print("No redundant tests found.")

if __name__ == "__main__":
    analyze_redundancy()
