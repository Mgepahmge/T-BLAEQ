#include <fstream>
#include <sstream>
#include <stdexcept>
#include "File.cuh"

PointCloud loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("loadFromFile: cannot open '" + filename + "'");
    }

    int length = 0, dim = 0;
    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("loadFromFile: empty file '" + filename + "'");
    }

    {
        std::istringstream iss(line);
        if (!(iss >> length >> dim) || length <= 0 || dim <= 0) {
            throw std::runtime_error("loadFromFile: invalid header in '" + filename + "'");
        }
    }

    PointCloud cloud(length, dim);

    int row = 0;
    while (std::getline(file, line) && row < length) {
        std::istringstream iss(line);
        double v;
        int col = 0;
        while (iss >> v && col < dim) {
            cloud.data[static_cast<size_t>(row) * dim + col++] = v;
        }

        if (col != dim) {
            throw std::runtime_error("loadFromFile: dimension mismatch at row " + std::to_string(row + 1));
        }
        ++row;
    }

    if (row != length) {
        throw std::runtime_error("loadFromFile: row count mismatch in '" + filename + "'");
    }

    return cloud;
}

void saveToFile(const PointCloud& cloud, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("saveToFile: cannot create '" + filename + "'");
    }

    file << cloud.size << ' ' << cloud.dim << '\n';
    for (int i = 0; i < cloud.size; ++i) {
        for (int j = 0; j < cloud.dim; ++j) {
            if (j) {
                file << ' ';
            }
            file << cloud.data[static_cast<size_t>(i) * cloud.dim + j];
        }
        file << '\n';
    }
}

std::vector<double> Query::getQueryPoint(int index) const {
    if (type != QueryType::POINT) {
        throw std::runtime_error("getQueryPoint: query type is not POINT");
    }
    if (index < 0 || index >= length) {
        throw std::out_of_range("getQueryPoint: index out of range");
    }

    std::vector<double> pt(static_cast<size_t>(dim));
    const size_t off = static_cast<size_t>(index) * dim;
    for (int j = 0; j < dim; ++j) {
        pt[j] = data[off + j];
    }
    return pt;
}

std::pair<std::vector<double>, std::vector<double>> Query::getQueryRange(int index) const {
    if (type != QueryType::RANGE) {
        throw std::runtime_error("getQueryRange: query type is not RANGE");
    }
    if (index < 0 || index >= length) {
        throw std::out_of_range("getQueryRange: index out of range");
    }

    std::vector<double> lo(static_cast<size_t>(dim));
    std::vector<double> hi(static_cast<size_t>(dim));
    const size_t off = static_cast<size_t>(index) * 2 * dim;
    for (int j = 0; j < dim; ++j) {
        lo[j] = data[off + j];
        hi[j] = data[off + dim + j];
    }
    return {lo, hi};
}

void Query::setQueryPoint(int index, const std::vector<double>& point) {
    if (type != QueryType::POINT) {
        throw std::runtime_error("setQueryPoint: query type is not POINT");
    }
    if (index < 0 || index >= length) {
        throw std::out_of_range("setQueryPoint: index out of range");
    }
    if (static_cast<int>(point.size()) != dim) {
        throw std::invalid_argument("setQueryPoint: dimension mismatch");
    }

    const size_t off = static_cast<size_t>(index) * dim;
    for (int j = 0; j < dim; ++j) {
        data[off + j] = point[j];
    }
}

void Query::setQueryRange(int index, const std::vector<double>& lower, const std::vector<double>& upper) {
    if (type != QueryType::RANGE) {
        throw std::runtime_error("setQueryRange: query type is not RANGE");
    }
    if (index < 0 || index >= length) {
        throw std::out_of_range("setQueryRange: index out of range");
    }
    if (static_cast<int>(lower.size()) != dim || static_cast<int>(upper.size()) != dim) {
        throw std::invalid_argument("setQueryRange: dimension mismatch");
    }

    const size_t off = static_cast<size_t>(index) * 2 * dim;
    for (int j = 0; j < dim; ++j) {
        data[off + j] = lower[j];
        data[off + dim + j] = upper[j];
    }
}

static Query loadQueryFromFile(const std::string& filename, QueryType expectedType) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("loadQueryFromFile: cannot open '" + filename + "'");
    }

    int length = 0, dim = 0;
    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("loadQueryFromFile: empty file");
    }

    {
        std::istringstream iss(line);
        if (!(iss >> length >> dim) || length <= 0 || dim <= 0) {
            throw std::runtime_error("loadQueryFromFile: invalid header");
        }
    }

    Query q(length, dim, expectedType);

    if (expectedType == QueryType::RANGE) {
        // File format: each row contains 2*dim values (lower bounds then upper bounds).
        // This matches the original format: data[i * 2*dim + 0..dim-1] = lower,
        //                                   data[i * 2*dim + dim..2*dim-1] = upper.
        const int expectedCols = 2 * dim;
        int rowRead = 0;
        while (std::getline(file, line) && rowRead < length) {
            std::istringstream iss(line);
            double v;
            int col = 0;
            while (iss >> v && col < expectedCols) {
                q.data[static_cast<size_t>(rowRead) * expectedCols + col++] = v;
            }

            if (col != expectedCols) {
                throw std::runtime_error("loadQueryFromFile: expected " + std::to_string(expectedCols) +
                                         " values but got " + std::to_string(col) + " at row " +
                                         std::to_string(rowRead + 1));
            }
            ++rowRead;
        }
        if (rowRead != length) {
            throw std::runtime_error("loadQueryFromFile: row count mismatch");
        }
    }
    else {
        // POINT query: each row contains dim values.
        int rowRead = 0;
        while (std::getline(file, line) && rowRead < length) {
            std::istringstream iss(line);
            double v;
            int col = 0;
            while (iss >> v && col < dim) {
                q.data[static_cast<size_t>(rowRead) * dim + col++] = v;
            }

            if (col != dim) {
                throw std::runtime_error("loadQueryFromFile: dimension mismatch at row " + std::to_string(rowRead + 1));
            }
            ++rowRead;
        }
        if (rowRead != length) {
            throw std::runtime_error("loadQueryFromFile: row count mismatch");
        }
    }

    return q;
}

Query loadQueryPointFromFile(const std::string& filename) { return loadQueryFromFile(filename, QueryType::POINT); }

Query loadQueryRangeFromFile(const std::string& filename) { return loadQueryFromFile(filename, QueryType::RANGE); }

static void saveQueryToFile(const Query& q, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("saveQueryToFile: cannot create '" + filename + "'");
    }

    file << q.length << ' ' << q.dim << '\n';

    if (q.type == QueryType::RANGE) {
        // Each row: 2*dim values (lower bounds then upper bounds)
        const int expectedCols = 2 * q.dim;
        for (int i = 0; i < q.length; ++i) {
            for (int j = 0; j < expectedCols; ++j) {
                if (j) {
                    file << ' ';
                }
                file << q.data[static_cast<size_t>(i) * expectedCols + j];
            }
            file << '\n';
        }
    }
    else {
        for (int i = 0; i < q.length; ++i) {
            for (int j = 0; j < q.dim; ++j) {
                if (j) {
                    file << ' ';
                }
                file << q.data[static_cast<size_t>(i) * q.dim + j];
            }
            file << '\n';
        }
    }
}

void saveQueryPointToFile(const Query& q, const std::string& filename) { saveQueryToFile(q, filename); }

void saveQueryRangeToFile(const Query& q, const std::string& filename) { saveQueryToFile(q, filename); }
