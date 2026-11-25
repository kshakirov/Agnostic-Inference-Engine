#include <iostream>


#include <iostream>
#include <Eigen/Dense>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

using Eigen::MatrixXd;
using namespace boost::accumulators;

int main() {
  MatrixXd m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << m << std::endl;

  accumulator_set<double, stats<tag::mean, tag::variance>> acc;

    // Добавляем числа (например, 8, 9, 10, 11, 12)
    acc(8);
    acc(9);
    acc(10);
    acc(11);
    acc(12);

    std::cout << "Mean: " << mean(acc) << std::endl;
    std::cout << "Variance: " << variance(acc) << std::endl;
}
