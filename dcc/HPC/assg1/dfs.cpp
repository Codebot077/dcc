#include <bits/stdc++.h> 
#include <ctime> 
#include <omp.h>
using namespace std;


using std :: chrono :: duration_cast;
using std :: chrono :: high_resolution_clock; 
using std :: chrono :: milliseconds;

void DFS(vector<vector<int>> &graph, vector<bool> &visited, int node, bool &print_node) {
    stack<int> s;
    s.push(node);

    #pragma omp parallel
    {
        #pragma omp single
        {
            while (!s.empty()) {
                int vertex;

                // Safely pop from stack
                #pragma omp critical
                {
                    if (!s.empty()) {
                        vertex = s.top();
                        s.pop();
                    } else {
                        vertex = -1;
                    }
                }

                if (vertex == -1) continue;

                // Only process if not visited
                bool already_visited = false;
                #pragma omp critical
                {
                    if (visited[vertex]) {
                        already_visited = true;
                    } else {
                        visited[vertex] = true;
                        if (print_node)
                            cout << vertex << " ";
                    }
                }

                if (already_visited) continue;

                // Create tasks for neighbors
                for (int neighbor : graph[vertex]) {
                    #pragma omp task firstprivate(neighbor)
                    {
                        bool local_visited;
                        #pragma omp critical
                        {
                            local_visited = visited[neighbor];
                            if (!local_visited)
                                s.push(neighbor);
                        }
                    }
                }
            }
        }
    }
}


void DFS_with_threads(vector<vector<int>> &graph, int start, bool &print_node){
    int N = graph.size();
    vector<bool> visited(N, false);
    DFS(graph, visited, start, print_node);
}

void DFS_without_threads(vector<vector<int>> &graph, vector<bool> & visited, int node, bool &print_node) {
    visited[node] = true;

    if(print_node)
        cout << node << " ";

    for(int i = 0; i < graph[node].size(); i++){
        if(!visited[graph[node][i]])
            DFS_without_threads(graph, visited, graph[node][i], print_node);
    }
}

void graph_input(vector<vector<int>> &graph){
    int N, choice = -1;
    cout<<"Enter the size of the graph : ";
    cin>>N;
    graph.resize(N);

    int total_edges;
    cout<<"Enter the no. of Edges : ";
    cin>>total_edges;

    for(int i = 0; i < total_edges; i++){
        int u, v;
        cout<<"Enter the current edge nodes named(0 to n-1): ";
        cin>>u>>v;

        if(u >= N || v >= N){
            cout<<"Nodes beyond the size of graph.\n";
            continue;
        }
        graph[u].push_back(v);
    }
}

int analysis(std :: function<void()> function){
    auto start = high_resolution_clock::now();
    function();
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);

    return duration.count();
}

int main(){
    vector<vector<int>> graph;
    vector<bool> visited;
    int N = 0;

    double execution1 = 0, execution2 = 0;
    bool print_node = false;

    int time_taken = 0;
    int num_of_vertices = 1000;
    int num_of_edges = 500000;
    float speed_up = 0.0f;

    bool flag = true;

    while(flag){
        cout<<"1. Sequential DFS \n";
        cout<<"2. Parallel DFS\n";
        cout<<"3. Compare Sequential and parellel DFS with random graph\n";
        cout<<"4. Exit\n";

        int choice = -1;
        cout << "Enter the choice : ";
        cin >> choice;

        switch(choice){
            case 1:
                graph_input(graph);
                print_node = true;

                N = graph.size();
                visited.resize(N, false);

                time_taken = analysis([&] {DFS_without_threads(graph, visited, 0, print_node);});
                cout << endl;
                cout << "Time taken : "<<time_taken << "\n";
            
                break;
            
            case 2:
                graph_input(graph);
                print_node = true;

                time_taken = analysis([&] {DFS_with_threads(graph, 0, print_node);});

                cout<<endl;
                cout<<"Time taken : "<<time_taken<<endl;

                break;

            case 3:
                graph.resize(num_of_vertices);
                for(int i = 0; i < num_of_edges; i++){
                    int u = (rand()% num_of_vertices);
                    int v = (rand()% num_of_vertices);

                    graph[u].push_back(v);
                    graph[v].push_back(u);
                }

                N = graph.size();
                visited.resize(N, false);

                print_node = false;
                execution1 = analysis([&] {DFS_without_threads(graph, visited, 0, print_node);});
                execution2 = analysis([&] {DFS_with_threads(graph, 0, print_node);});

                cout << "Sequential time : "<< execution1 << "ms" << endl;
                cout << "Parellel time : "<< execution2 <<"ms"<<endl;
                cout << "Speed Up : "<<speed_up<<endl;

                graph.clear();
                break;

            case 4:
                flag = false;
                break;

            default:

                cout<<"Invalid Input " << endl;
                break;
        }
    }
    return 0;
}


/*
To run this
1. compile using : g++ -fopenmp dfs.cpp
2. run using     : ./a.out
*/


/*
To run this
1. compile using : g++ -fopenmp dfs.cpp
2. run using     : ./a.out
*/


#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using namespace std::chrono;

void DFS_parallel(vector<vector<int>>& graph, vector<bool>& visited, int node) {
    stack<int> s;
    s.push(node);

    #pragma omp parallel
    {
        #pragma omp single
        {
            while (!s.empty()) {
                int vertex = -1;

                #pragma omp critical
                {
                    if (!s.empty()) {
                        vertex = s.top();
                        s.pop();
                    }
                }

                if (vertex == -1) continue;

                bool already_visited = false;

                #pragma omp critical
                {
                    if (visited[vertex]) {
                        already_visited = true;
                    } else {
                        visited[vertex] = true;
                    }
                }

                if (already_visited) continue;

                for (int neighbor : graph[vertex]) {
                    #pragma omp task firstprivate(neighbor)
                    {
                        #pragma omp critical
                        {
                            if (!visited[neighbor])
                                s.push(neighbor);
                        }
                    }
                }
            }
        }
    }
}

void DFS_sequential(vector<vector<int>>& graph, vector<bool>& visited, int node) {
    visited[node] = true;
    for (int neighbor : graph[node]) {
        if (!visited[neighbor])
            DFS_sequential(graph, visited, neighbor);
    }
}

vector<vector<int>> generate_random_graph(int vertices, int edges) {
    vector<vector<int>> graph(vertices);
    for (int i = 0; i < edges; i++) {
        int u = rand() % vertices;
        int v = rand() % vertices;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }
    return graph;
}

int main() {
    int vertices = 1000, edges = 500000;

    auto graph = generate_random_graph(vertices, edges);

    vector<bool> visited_seq(vertices, false);
    vector<bool> visited_par(vertices, false);

    auto start_seq = high_resolution_clock::now();
    DFS_sequential(graph, visited_seq, 0);
    auto end_seq = high_resolution_clock::now();

    auto start_par = high_resolution_clock::now();
    DFS_parallel(graph, visited_par, 0);
    auto end_par = high_resolution_clock::now();

    auto t_seq = duration_cast<milliseconds>(end_seq - start_seq).count();
    auto t_par = duration_cast<milliseconds>(end_par - start_par).count();

    cout << "Sequential DFS: " << t_seq << " ms\n";
    cout << "Parallel DFS: " << t_par << " ms\n";
    cout << "Speedup: " << fixed << setprecision(2) << (float)t_seq / t_par << "x\n";

    return 0;
}
