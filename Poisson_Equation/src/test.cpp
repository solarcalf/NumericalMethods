#include <vector>
#include <algorithm>
#include <memory>
#include <omp.h>
#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include "../headers/Dirichlet_Problem.hpp"
#include "../headers/Solvers.hpp"
#include "../headers/tasks.hpp"

#define FP double

template<numcpp::GridType Grid>
std::pair<size_t, std::vector<std::vector<FP>>> estimate_time(numcpp::DirichletProblemSolver<Grid>& solver)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto solution = solver.solve();
    auto end = std::chrono::high_resolution_clock::now();

    int64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Overall time: " << duration / 1000000 << "s\n";

    return {duration, solution};
}

void report(size_t task_num, size_t n, size_t m, size_t max_iterations, FP eps)
{
    std::vector<std::string> results;
    std::vector<std::string> results_err;
    std::string line = "";
    std::string curr_string = "";
    std::string start_string = "";

    std::ifstream solver_results("../files/Solver_results.txt");
    while (std::getline(solver_results, line)) 
        results.push_back(line);
    solver_results.close();

    std::ifstream error("../files/Error.txt");
    while (std::getline(error, line)) 
        results_err.push_back(line);
    error.close();


    
    if(task_num == 0)
    {
        start_string = "Для решения тестовой задачи использованы сетка ";
    }
    else if(task_num == 1)
    {
        start_string = "Для решения основной задачи использованы сетка ";

        curr_string = "\nДля контроля точности использована сетка(2N) \n с числом разбиений по x : n = ";
        curr_string += std::to_string(n * 2);
        curr_string += ", и числом разбиений по y: m = ";
        curr_string += std::to_string(m * 2);
        curr_string += ". \nКритерии остановки метода остаются такими же.";

        std::ifstream solver_results_2("../files/Solver_results_2N.txt");
        std::vector<std::string> results_2;
        while (std::getline(solver_results_2, line))
            results_2.push_back(line);
        solver_results_2.close();


        curr_string += "На решение СЛАУ(2N) затрачено ";
        curr_string += results_2[2];
        curr_string += " итераций\n и достигнута точность ";
        curr_string += results_2[0];
        curr_string += ".СЛАУ(2N) решена с невязкой по норме Чебышёва ";
        curr_string += results_2[1];
    }
    else
    {
        start_string = "Для решения тестовой задачи использованы сетка (нестандартная сетка) ";
    }

    std::cout << "\n\n\n\n" << start_string << " с числом разбиений\n по х : n = " << n
            << ", и числом разбиений по y: m = " << m << ",\n "
            << "применены критерии остановки \n по точности решения СЛАУ eps(мет) = " << eps
            << " и по числу итераций N(max) = " << max_iterations << ".\n \n "
            << "На решение схемы (СЛАУ) затрачено " << results[2]
            << " итераций и достигнута точность " << results[0]
            << "\nСхема(СЛАУ) решена с невязкой по норме Чебышёва " << results[1] << "\n" << results_err[0] << "\n" << results_err[1] << "\n\n" << curr_string << "\n\n\n";


}


// for test we use functions from tasks.hpp

int main() 
{
    size_t m = 100;
    size_t n = 100;
    // solver_num chooses a method for solving a system of linear equations
    // 0 = TopRelaxation
    // 1 = MinRes
    // 2 = ChebyshevIteration
    // 3 = ConGrad
    size_t solver_num = 3;
    size_t max_iterations = 1000000; 
    FP eps = 0.000001;
    FP omega = 1.5;
    // task_num 
    // 0 = test 
    // 1 = main 
    // 2 = custom test
    size_t task_num = 1;

    //test_task(solver_num, n, m, max_iterations, eps, omega); // Regular grid
    main_task(solver_num, n, m, max_iterations, eps, omega); // Regular grid
    // test_custom_task(solver_num, n, m, max_iterations, eps, omega); //ReversedR grid

    // the values in the grid nodes can be found in the files
    // for approximation in file "../files/Approximation.txt"
    // for correct solution in file "../files/Correct.txt"

    // report in console
    report(task_num, n, m, max_iterations, eps);

}