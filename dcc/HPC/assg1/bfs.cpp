// #include <bits/stdc++.h> 
// #include <ctime> 
// #include <omp.h>
// using namespace std;

// using std :: chrono :: duration_cast;
// using std :: chrono :: high_resolution_clock; 
// using std :: chrono :: milliseconds;


// void BFS(vector<vector<int>> &graph, int node, vector<bool>& visited, bool &print_node) {

//     queue<int> Queue;
// 	Queue.push(node);
// 	visited[node]=true;
	
// 	#pragma omp parallel
// 	{
//         #pragma omp single
//         {
//             while (!Queue.empty()){
//                 int vertex = Queue.front();
//                 Queue.pop();

//                 if (print_node)
//                     cout<<vertex<<" ";
	
	
//                 #pragma omp task firstprivate(vertex)
//                 {
//                     for(int neighbor : graph[vertex]){
//                         if(!visited[neighbor]){
//                             Queue.push(neighbor);
//                             visited[neighbor] = true;

//                             #pragma omp task
//                             BFS(graph, neighbor, visited, print_node);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }
	
// void parellel_BFS(vector<vector<int>>&graph, int start, bool &print_node){
//     int N = graph.size();
//     vector<bool> visited(N, false);
//     BFS(graph, start, visited,print_node);
// }

// void sequential_BFS(vector<vector<int>> &graph, int start_node, bool& print_node){
//     int N = graph.size();
//     vector<bool> visited(N, false);
//     queue<int> Queue;
//     Queue.push(start_node);
//     visited[start_node] = true;

//     while(!Queue.empty()){
//         int vertex = Queue.front();

//         if(print_node)
//             cout<<vertex<<" ";

//         Queue.pop();

//         for(int neighbor : graph[vertex]){
//             if(!visited[neighbor]){
//                 Queue.push(neighbor);
//                 visited[neighbor] = true;
//             }
//         }
//     }
// }

// void graph_input(vector<vector<int>> &graph){
//     int N, choice = -1;
//     cout<<"Enter the size of the graph : ";
//     cin>>N;
//     graph.resize(N);

//     int total_edges;
//     cout<<"Enter the no. of Edges : ";
//     cin>>total_edges;

//     for(int i = 0; i < total_edges; i++){
//         int u, v;
//         cout<<"Enter the current edge nodes named(0 to n-1): ";
//         cin>>u>>v;

//         if(u >= N || v >= N){
//             cout<<"Nodes beyond the size of graph.\n";
//             continue;
//         }
//         graph[u].push_back(v);
//         graph[v].push_back(u);
//     }
// }

// int analysis(std :: function<void()> function){
//     auto start = high_resolution_clock::now();
//     function();
//     auto end = high_resolution_clock::now();

//     auto duration = duration_cast<milliseconds>(end - start);

//     return duration.count();
// }

// int main(void){
//     vector<vector<int>> graph;
//     vector<bool> visited;

//     double sequential_execution = 0, parellel_execution = 0;
//     bool print_node = false;

//     int time_taken = 0;
//     int num_of_vertices = 1000;
//     int num_of_edges = 500000;
//     float speed_up = 0.0f;
//     bool flag = true;

//     while(flag){
//         cout<<"1. Sequential BFS \n";
//         cout<<"2. Parellel BFS\n";
//         cout<<"3. Compare Sequential and parellel BFS with random graph\n";
//         cout<<"4. Exit\n";

//         int choice = -1;
//         cout << "Enter the choice : ";
//         cin >> choice;

//         switch(choice){
//             case 1:
//                 graph_input(graph);
//                 print_node = true;

//                 time_taken = analysis([&] {sequential_BFS(graph, 0, print_node);});

//                 cout << endl;
//                 cout << "Time Taken : " << time_taken << endl;
//                 break;

//             case 2:
//                 graph_input(graph);
//                 print_node = true;

//                 time_taken = analysis([&] {parellel_BFS(graph, 0, print_node);});

//                 cout << endl;
//                 cout << "Time Taken : " << time_taken << endl;

//                 break;

//             case 3:
//                 graph.resize(num_of_vertices);

//                 for(int i = 0; i < num_of_edges; i++){
//                     int u = (rand() % num_of_vertices);
//                     int v = (rand() % num_of_vertices);

//                     graph[u].push_back(v);
//                     graph[v].push_back(u);
//                 }
//                 print_node = false;
//                 sequential_execution = analysis([&] {sequential_BFS(graph, 0, print_node);});
//                 parellel_execution = analysis([&] {parellel_BFS(graph, 0, print_node);});

//                 speed_up = (float) (sequential_execution/ (float) parellel_execution);

//                 cout << "Sequential time : "<<sequential_execution<<"ms\n";
//                 cout << "Parellel time : "<<parellel_execution<<"ms\n";
//                 cout << "Speed Up : "<<speed_up<<"\n";
//                 break;
//             case 4:
//                 flag = false;
//                 break;
//             default:
//                 cout << "Invalid Input \n";
//                 break;
//         }
//     }
//     return 0;
// }

/*
To run this
1. compile using : g++ -fopenmp bfs.cpp
2. run using     : ./a.out
*/





#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <functional>
#include <omp.h>
using namespace std;
using namespace std::chrono;

void sequential_BFS(const vector<vector<int>> &graph, int start_node) {
    int N = graph.size();
    vector<bool> visited(N, false);
    queue<int> q;
    q.push(start_node);
    visited[start_node] = true;

    while (!q.empty()) {
        int node = q.front(); q.pop();
    
        for (int neighbor : graph[node]) {
            
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);  
            }
        }
    }
}


// Simplified & fixed "parallel BFS" (level-based parallelism)
void parallel_BFS(const vector<vector<int>> &graph, int start_node) {
    int N = graph.size();
    vector<bool> visited(N, false);
    queue<int> q;
    visited[start_node] = true;
    q.push(start_node);

    while (!q.empty()) {
        int sz = q.size();
        vector<int> current_level;

        while (sz--) {
            int node = q.front(); q.pop();
            current_level.push_back(node);
        }

        #pragma omp parallel for
        for (int i = 0; i < current_level.size(); ++i) {
            int node = current_level[i];
            for (int neighbor : graph[node]) {
                #pragma omp critical
                {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }
    }
}


int analyze(function<void()> func) {
    auto start = high_resolution_clock::now();
    func();
    auto end = high_resolution_clock::now();
    return duration_cast<milliseconds>(end - start).count();
}


int main() {
    const int V = 1000, E = 500000;
    vector<vector<int>> graph(V);

    // Random graph
    for (int i = 0; i < E; ++i) {
        int u = rand() % V;
        int v = rand() % V;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    int t1 = analyze([&]() { sequential_BFS(graph, 0); });
    int t2 = analyze([&]() { parallel_BFS(graph, 0); });

    cout << "Sequential BFS: " << t1 << "ms\n";
    cout << "Parallel BFS:   " << t2 << "ms\n";
    cout << "Speedup: " << (float)t1 / t2 << "x\n";

    return 0;
}


/*
To run this
1. compile using : g++ -fopenmp bfs.cpp
2. run using     : ./a.out
*/
