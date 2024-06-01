#include <vector>
#include <algorithm>
#include <memory>
#include <omp.h>
#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>

#include "../headers/Dirichlet_Problem.hpp"
#include "../headers/Solvers.hpp"

#define FP double


void test_task(size_t solver_num, size_t n, size_t m, size_t max_iterations, FP eps, FP omega){
    std::array<double, 4> corners = {-1.0, -1.0, 1.0, 1.0};
    std::vector<FP> init_app((n - 1) * (m - 1), 0.0);

    auto u = [](double x, double y) { return exp(1 - pow(x, 2) - pow(y, 2)); };
    auto f = [](double x, double y) { return -4 * exp(1 - pow(x, 2) - pow(y, 2)) * (pow(y, 2) + pow(x, 2) - 1); };

    auto mu1 = [](double y) { return exp(-pow(y, 2)); };
    auto mu2 = [](double y) { return exp(-pow(y, 2)); };
    auto mu3 = [](double x) { return exp(-pow(x, 2)); };
    auto mu4 = [](double x) { return exp(-pow(x, 2)); };
    numcpp::DirichletProblemSolver<numcpp::GridType::Regular> dirichlet_task;
    
    dirichlet_task.set_fraction(m, n);
    dirichlet_task.set_corners(corners);
    dirichlet_task.set_u(u);
    dirichlet_task.set_f(f);
    dirichlet_task.set_task_num(0);
    dirichlet_task.set_boundary_conditions({mu1, mu2, mu3, mu4});

    FP h = (corners[2] - corners[0]) / n;
    FP k = (corners[3] - corners[1]) / m;
    FP Mmin = 4.0 / pow(h, 2) * pow(sin(numcpp::PI / 2.0 / n), 2) + 4.0 / pow(k, 2) * pow(sin(numcpp::PI / 2.0 / m), 2);
    FP Mmax = 4.0 / pow(h, 2) * pow(sin(numcpp::PI * (n - 1) / 2.0 / n), 2) + 4.0 / pow(k, 2) * pow(sin(numcpp::PI * (m - 1) / 2.0 / m), 2);

    if(solver_num == 0)
        dirichlet_task.set_solver(std::make_unique<numcpp::TopRelaxationOptimizedForDirichletRegularGrid>
        (init_app, max_iterations, eps, nullptr, std::vector<FP>(), f, mu1, mu2, mu3, mu4, n, m, corners, omega));
    else if(solver_num == 1)
        dirichlet_task.set_solver(std::make_unique<numcpp::MinRes>(init_app, max_iterations, eps, nullptr, std::vector<FP>()));
    else if (solver_num == 2){
        Mmin*=-1;
        Mmax*=-1;
        dirichlet_task.set_solver(std::make_unique<numcpp::ChebyshevIteration>(init_app, max_iterations, eps, nullptr, std::vector<FP>(), Mmin, Mmax));
    }
    else 
        dirichlet_task.set_solver(std::make_unique<numcpp::ConGrad>(init_app, max_iterations, eps, nullptr, std::vector<FP>()));

    auto start = std::chrono::high_resolution_clock::now();
    auto solution = dirichlet_task.solve();
    auto end = std::chrono::high_resolution_clock::now();

    int64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Overall time: " << duration / 1000000 << "s\n";

    std::ofstream fapp("../files/Approximation.txt");
    std::ofstream fcurr("../files/Correct.txt");
    for(size_t j = m; j > 0; j--)
    {
        for(size_t i = 0; i <= n; i++)
        {
            FP x = corners[0] + i * h;
            FP y = corners[1] + j* k;
            fapp << solution[i][j] <<' ';
            fcurr << u(x, y) << ' ';
        }
        fapp << '\n';
        fcurr << '\n';
    }
    fapp.close();
    fcurr.close();


}


void test_custom_task(size_t solver_num, size_t n, size_t m, size_t max_iterations, FP eps, FP omega){
    std::array<double, 4> corners = {-1.0, -1.0, 1.0, 1.0};
    size_t size = (n / 2 - 1) * (m - 1) + (n / 2) * (m / 2 - 1);
    std::vector<FP> init_app(size, 0.0);

    auto u = [](double x, double y) { return exp(1 - pow(x, 2) - pow(y, 2)); };
    auto f = [](double x, double y) { return -4 * exp(1 - pow(x, 2) - pow(y, 2)) * (pow(y, 2) + pow(x, 2) - 1); };

    auto mu1 = [](double y) { return exp(-pow(y, 2)); };
    auto mu2 = [](double y) { return exp(1.0 - pow(y, 2)); };
    auto mu3 = [](double y) { return exp(-pow(y, 2)); };
    auto mu4 = [](double x) { return exp(-pow(x, 2)); };
    auto mu5 = [](double x) { return exp(1.0 - pow(x, 2)); };
    auto mu6 = [](double x) { return exp(-pow(x, 2)); };

    numcpp::DirichletProblemSolver<numcpp::GridType::ReversedR> dirichlet_task;
    std::array<std::function<FP(FP)>, 6> arr{mu1, mu2, mu3, mu4, mu5, mu6};
    
    dirichlet_task.set_fraction(m, n);
    dirichlet_task.set_corners(corners);
    dirichlet_task.set_u(u);
    dirichlet_task.set_f(f);
    dirichlet_task.set_boundary_conditions_for_r_shaped_grid(arr);

    FP h = (corners[2] - corners[0]) / n;
    FP k = (corners[3] - corners[1]) / m;

    FP Mmin = 4.0 / pow(h, 2) * pow(sin(numcpp::PI / 2.0 / n), 2) + 4.0 / pow(k, 2) * pow(sin(numcpp::PI / 2.0 / m), 2);
    FP Mmax = 4.0 / pow(h, 2) * pow(sin(numcpp::PI * (n - 1) / 2.0 / n), 2) + 4.0 / pow(k, 2) * pow(sin(numcpp::PI * (m - 1) / 2.0 / m), 2);

    if(solver_num == 0)
        dirichlet_task.set_solver(std::make_unique<numcpp::TopRelaxation>(init_app, max_iterations, eps, nullptr, std::vector<FP>(), omega));
    else if(solver_num == 1)
        dirichlet_task.set_solver(std::make_unique<numcpp::MinRes>(init_app, max_iterations, eps, nullptr, std::vector<FP>()));
    else if (solver_num == 2)
        dirichlet_task.set_solver(std::make_unique<numcpp::ChebyshevIteration>(init_app, max_iterations, eps, nullptr, std::vector<FP>(), Mmin, Mmax));
    else 
        dirichlet_task.set_solver(std::make_unique<numcpp::ConGrad>(init_app, max_iterations, eps, nullptr, std::vector<FP>()));

    auto start = std::chrono::high_resolution_clock::now();
    auto solution = dirichlet_task.solve();
    auto end = std::chrono::high_resolution_clock::now();

    int64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Overall time: " << duration / 1000000 << "s\n";

    std::ofstream fapp("../files/Approximation.txt");
    std::ofstream fcurr("../files/Correct.txt");
    
    for(size_t j = m; j > 0; j--)
    {
        for(size_t i = 0; i <= n; i++)
        {
            FP x = corners[0] + i * h;
            FP y = corners[1] + j * k;
            fapp << solution[i][j] <<' ';
            if(i < n / 2 && j < m / 2)
                fcurr << 0.0 << ' ';
            else
                fcurr << u(x, y) << ' ';
        }
        fapp << '\n';
        fcurr << '\n';
    }
    fapp.close();
    fcurr.close();
}


void main_task(size_t solver_num, size_t n, size_t m, size_t max_iterations, FP eps, FP omega){
    std::array<double, 4> corners = {-1.0, -1.0, 1.0, 1.0};
    std::vector<FP> init_app((n - 1) * (m - 1), 0.0);

    auto f = [](double x, double y) { return std::abs(pow(sin(numcpp::PI*x*y), 3)); };

    auto mu1 = [](double y) { return -1*pow(y, 2)+1; };
    auto mu2 = [](double y) { return -1*pow(y, 2)+1; };
    auto mu3 = [](double x) { return -1*pow(x, 2)+1; };
    auto mu4 = [](double x) { return -1*pow(x, 2)+1; };
    numcpp::DirichletProblemSolver<numcpp::GridType::Regular> dirichlet_task;
    
    dirichlet_task.set_fraction(m, n);
    dirichlet_task.set_corners(corners);
    dirichlet_task.set_f(f);
    dirichlet_task.set_task_num(1);
    dirichlet_task.set_boundary_conditions({mu1, mu2, mu3, mu4});

    FP h = (corners[2] - corners[0]) / n;
    FP k = (corners[3] - corners[1]) / m;
    FP Mmin = 4.0 / pow(h, 2) * pow(sin(numcpp::PI / 2.0 / n), 2) + 4.0 / pow(k, 2) * pow(sin(numcpp::PI / 2.0 / m), 2);
    FP Mmax = 4.0 / pow(h, 2) * pow(sin(numcpp::PI * (n - 1) / 2.0 / n), 2) + 4.0 / pow(k, 2) * pow(sin(numcpp::PI * (m - 1) / 2.0 / m), 2);

    if(solver_num == 0)
        dirichlet_task.set_solver(std::make_unique<numcpp::TopRelaxationOptimizedForDirichletRegularGrid>
        (init_app, max_iterations, eps, nullptr, std::vector<FP>(), f, mu1, mu2, mu3, mu4, n, m, corners, omega));
    else if(solver_num == 1)
        dirichlet_task.set_solver(std::make_unique<numcpp::MinRes>(init_app, max_iterations, eps, nullptr, std::vector<FP>()));
    else if (solver_num == 2)
        dirichlet_task.set_solver(std::make_unique<numcpp::ChebyshevIteration>(init_app, max_iterations, eps, nullptr, std::vector<FP>(), Mmin, Mmax));
    else 
        dirichlet_task.set_solver(std::make_unique<numcpp::ConGrad>(init_app, max_iterations, eps, nullptr, std::vector<FP>()));

    auto start = std::chrono::high_resolution_clock::now();
    auto solution = dirichlet_task.solve();
    auto end = std::chrono::high_resolution_clock::now();

    int64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Overall time first: " << duration / 1000000 << "s\n";

    std::vector<std::string> lines;
    std::ifstream file("../files/Solver_results.txt");
    if (file.is_open()) 
    {
        std::string line;
        while (getline(file, line)) 
            lines.push_back(line);
        file.close();
    }

    n *= 2;
    m *= 2;

    std::vector<FP> init_app_2((n - 1) * (m - 1), 0.0);
    numcpp::DirichletProblemSolver<numcpp::GridType::Regular> dirichlet_task_2;
    
    dirichlet_task_2.set_fraction(m, n);
    dirichlet_task_2.set_corners(corners);
    dirichlet_task_2.set_f(f);
    dirichlet_task_2.set_task_num(1);
    dirichlet_task_2.set_boundary_conditions({mu1, mu2, mu3, mu4});

    h = (corners[2] - corners[0]) / n;
    k = (corners[3] - corners[1]) / m;
    Mmin = 4.0 / pow(h, 2) * pow(sin(numcpp::PI / 2.0 / n), 2) + 4.0 / pow(k, 2) * pow(sin(numcpp::PI / 2.0 / m), 2);
    Mmax = 4.0 / pow(h, 2) * pow(sin(numcpp::PI * (n - 1) / 2.0 / n), 2) + 4.0 / pow(k, 2) * pow(sin(numcpp::PI * (m - 1) / 2.0 / m), 2);

    if(solver_num == 0)
        dirichlet_task_2.set_solver(std::make_unique<numcpp::TopRelaxationOptimizedForDirichletRegularGrid>
        (init_app_2, max_iterations, eps, nullptr, std::vector<FP>(), f, mu1, mu2, mu3, mu4, n, m, corners, omega));
    else if(solver_num == 1)
        dirichlet_task_2.set_solver(std::make_unique<numcpp::MinRes>(init_app_2, max_iterations, eps, nullptr, std::vector<FP>()));
    else if (solver_num == 2)
        dirichlet_task_2.set_solver(std::make_unique<numcpp::ChebyshevIteration>(init_app_2, max_iterations, eps, nullptr, std::vector<FP>(), Mmin, Mmax));
    else 
        dirichlet_task_2.set_solver(std::make_unique<numcpp::ConGrad>(init_app_2, max_iterations, eps, nullptr, std::vector<FP>()));

    auto start_2 = std::chrono::high_resolution_clock::now();
    auto solution_2 = dirichlet_task_2.solve();
    auto end_2 = std::chrono::high_resolution_clock::now();

    auto duration_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2).count();
    std::cout << "Overall time second: " << duration_2 / 1000000 << "s\n";

    FP error = 0; 
    FP x_max = 0;
    FP y_max = 0;

    n /= 2;
    m /= 2;

    std::ofstream fapp("../files/Approximation.txt");
    std::ofstream fcurr("../files/Correct.txt");

    for(size_t j = m; j > 0; j--)
    {
        for(size_t i = 0; i <= n; i++)
        {
            FP y = corners[1] + i * k*2;
            FP x = corners[0] + j * h*2;
            FP dot_solution = solution[i][j];
            FP dot_correct =  solution_2[i*2][j*2];
            if(std::abs(dot_correct - dot_solution) > error)
            {
                error = std::abs(dot_correct - dot_solution);
                x_max = x;
                y_max = y;
            }
            fapp << dot_solution <<' ';
            fcurr << dot_correct << ' ';
        } 
        fapp << '\n';
        fcurr << '\n';
    }
    fapp.close();
    fcurr.close();

    std::ofstream fout_res("../files/Error.txt");
    fout_res << "Задача решена с точностью " << error << "\nМаксимальное отклонение 'точного' и численного решений в точке x = " << x_max << ", y = " << y_max << '\n';
    fout_res.close();

    std::vector<std::string> lines_2;
    std::ifstream file_2("../files/Solver_results.txt");
    if (file_2.is_open()) 
    {
        std::string line;
        while (getline(file_2, line)) 
            lines_2.push_back(line);
        file_2.close();
    }

    std::ofstream fout("../files/Solver_results.txt");
    fout << lines[0] << '\n' << lines[1] << '\n' << lines[2];
    fout.close();

    std::ofstream fout_2("../files/Solver_results_2N.txt");
    fout_2 << lines_2[0] << '\n' << lines_2[1] << '\n' << lines_2[2];
    fout_2.close();
}
