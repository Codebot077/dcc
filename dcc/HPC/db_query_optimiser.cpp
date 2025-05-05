#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <chrono>
using namespace std;

// Employee table row
struct Employee
{
    int emp_id;
    string name;
    int dept_id;
};

// Department table row
struct Department
{
    int dept_id;
    string dept_name;
};

int main()
{
    vector<Employee> employees = {
        {1, "Robin", 102},
        {2, "Nami", 103},
        {3, "Usopp", 104},
        {4, "Chopper", 101},
        {5, "Ace", 102},
        {6, "Jinbe", 104},
        {7, "Sabo", 103},
        {8, "Killua", 101},
        {9, "Yamato", 102},
        {10, "Law", 105}};

    vector<Department> departments = {
        {101, "Support"},
        {102, "Engg"},
        {103, "HR"},
        {104, "Content-writing"},
        {105, "Security"}    };

    vector<pair<string, string>> result;

    auto start = chrono::high_resolution_clock::now();

    // Filtered departments (WHERE dept_name = 'Support')
    vector<int> engineeringDeptIds;

#pragma omp parallel for
    for (int i = 0; i < departments.size(); i++)
    {
        if (departments[i].dept_name == "Support")
        {
#pragma omp critical
            engineeringDeptIds.push_back(departments[i].dept_id);
        }
    }

// Join operation (parallel)
#pragma omp parallel for
    for (int i = 0; i < employees.size(); i++)
    {
        for (int j = 0; j < engineeringDeptIds.size(); j++)
        {
            if (employees[i].dept_id == engineeringDeptIds[j])
            {
                string empName = employees[i].name;
                string deptName = "Support"; // already filtered

#pragma omp critical
                result.push_back({empName, deptName});
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    // printing the database
    cout << "Employee database" << endl;
    cout << "Empi_id \t Name \t\t Dept_id\n";
    for (auto it : employees)
    {
        cout << it.emp_id << " \t\t " << it.name << " \t\t " << it.dept_id << endl;
    }
    cout << endl;

    cout << "Departments database" << endl;
    cout << "Dept_id \t Dept_name\n";
    for (auto it : departments)
    {
        cout << it.dept_id << " \t\t " << it.dept_name << endl;
    }
    cout << endl;

    // Output
    cout << "Optimized Query Result:\n";
    for (auto &row : result)
    {
        cout << row.first << " works in " << row.second << endl;
    }

    cout << "\nTotal execution time: " << duration.count() << " seconds\n";

    return 0;
}

/*
To run this
1. compile using : g++ -fopenmp db_query_optimiser.cpp
2. run using     : ./a.out
*/
