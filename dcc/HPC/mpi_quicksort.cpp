#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

// Standard quicksort for local sorting
void quicksort(std::vector<int>& arr, int left, int right) {
    if (left >= right) return;

    int pivot = arr[right]; // Choosing last element as pivot
    int i = left - 1;

    for (int j = left; j < right; ++j) {
        if (arr[j] <= pivot) {
            std::swap(arr[++i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[right]);

    // Recursive calls for left and right partitions
    quicksort(arr, left, i);
    quicksort(arr, i + 2, right);
}

// Merges two sorted vectors into one
std::vector<int> merge(const std::vector<int>& left, const std::vector<int>& right) {
    std::vector<int> result;
    int i = 0, j = 0;

    while (i < left.size() && j < right.size()) {
        if (left[i] <= right[j]) result.push_back(left[i++]);
        else result.push_back(right[j++]);
    }

    // Add remaining elements
    while (i < left.size()) result.push_back(left[i++]);
    while (j < right.size()) result.push_back(right[j++]);

    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Initialize MPI

    int rank, size;
    const int N = 1000; // Total number of elements to sort

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    int chunk_size = N / size; // Divide work equally among processes
    std::vector<int> data;
    std::vector<int> local_data(chunk_size);

    double start_time, end_time;

    if (rank == 0) {
        // Only master initializes full data array
        data.resize(N);
        srand(time(0));
        for (int i = 0; i < N; ++i) {
            data[i] = rand() % 10000; // Random integers
        }

        std::cout << "Unsorted array (starting 30 elements): ";
        for (int i = 0; i < 30; ++i) std::cout << data[i] << " ";
        std::cout << "\n";

        // Start timing from here
        start_time = MPI_Wtime();
    }

    // Distribute data equally to all processes
    MPI_Scatter(data.data(), chunk_size, MPI_INT, local_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process sorts its local chunk
    quicksort(local_data, 0, chunk_size - 1);

    // Gather all sorted chunks at root process
    std::vector<int> gathered_data;
    if (rank == 0) {
        gathered_data.resize(N);
    }

    MPI_Gather(local_data.data(), chunk_size, MPI_INT, gathered_data.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Merge all sorted chunks
        std::vector<int> result(gathered_data.begin(), gathered_data.begin() + chunk_size);
        for (int i = 1; i < size; ++i) {
            std::vector<int> next_chunk(gathered_data.begin() + i * chunk_size,
                                        gathered_data.begin() + (i + 1) * chunk_size);
            result = merge(result, next_chunk);
        }

        // End timing after merge
        end_time = MPI_Wtime();

        std::cout << "\nSorted array (starting 30 elements) : ";
        for (int i = 0; i < 30; ++i) std::cout << result[i] << " ";
        std::cout << "\n";

        std::cout << "\nTotal execution time: " << (end_time - start_time) << " seconds\n";
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}

/*
To run this
1. compile using : mpic++ mpi_quicksort.cpp -o output_file_name
2. run using     : mpirun -np 2 ./output_file_name
    The -np flag stands for "number of processes".
    It tells MPI how many parallel processes you want to spawn for your program.
*/
