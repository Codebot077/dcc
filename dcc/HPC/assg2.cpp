#include <bits/stdc++.h>
#include<omp.h>
using namespace std;
void bubble(int array[], int n){
    for (int i = 0; i < n - 1; i++){
        for (int j = 0; j < n - i - 1; j++){
            if (array[j] > array[j + 1]) swap(array[j], array[j + 1]);
        }
    }
}
void parallel_bubblesort(int array[], int n) {
    for (int i = 0; i < n; i++)
    {
        int first = i % 2;
        // defining the shared data source for parallel execution.
        #pragma omp parallel for shared (array, first) num_threads(16)
        for (int j = first; j < n - 1; j += 2)
            if (array[j] > array[j + 1])
                swap(array[j], array[j + 1]);
    }
    return;
}
void merge(int arr[], int low, int mid, int high) {
    // Create arrays of left and right partititons
    int n1 = mid - low + 1;
    int n2 = high - mid;

    int left[n1];
    int right[n2];
    
    // Copy all left elements
    for (int i = 0; i < n1; i++) left[i] = arr[low + i];
    
    // Copy all right elements
    for (int j = 0; j < n2; j++) right[j] = arr[mid + 1 + j];
    
    // Compare and place elements
    int i = 0, j = 0, k = low;

    while (i < n1 && j < n2) {
        if (left[i] <= right[j]){
            arr[k] = left[i];
            i++;
        } 
        else{
            arr[k] = right[j];
            j++;
        }
        k++;
    }

    // If any elements are left out
    while (i < n1) {
        arr[k] = left[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = right[j];
        j++;
        k++;
    }
}
void mergeSort(int arr[], int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;
        mergeSort(arr, low, mid);
        mergeSort(arr, mid + 1, high);
        merge(arr, low, mid, high);
    }
}

void parallelMergeSort(int arr[], int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                mergeSort(arr, low, mid);
            }

            #pragma omp section
            {
                mergeSort(arr, mid + 1, high);
            }
        }
        merge(arr, low, mid, high);
    }
}
int main(){
    // Set up variables
    int n = 100000;
    int arr[n];
    int brr[n],crr[n],drr[n];
    double start_time, end_time;
    // Create an array with random numbers
    for(int i = 0, j = n; i < n; i++, j--){
    		arr[i] = 1 + (rand() % 1000);
    		brr[i]=arr[i];
            crr[i]=arr[i];
            drr[i]=arr[i];
    }
    bool flag = true;
    while(flag){
        cout << "-------------------------- Menu -------------------------------\n";
        cout << "1. Bubble Sort   \n" ;
        cout << "2. Merge Sort \n" << endl;
        cout << "3. Exit \n";
        cout << "---------------------------------------------------------------\n";
        int choice =-1;
        cout<<"Enter your choice: ";
        cin>>choice;
        switch(choice)
        {
            case 1:
                cout << "Bubble Sort\n";
                // Sequential time
                start_time = omp_get_wtime();
                bubble(arr, n);
                end_time = omp_get_wtime();     
                cout << "Sequential Bubble Sort time = " << end_time - start_time << " seconds.\n";
                
                // Parallel time
                start_time = omp_get_wtime();
                parallel_bubblesort(brr, n);
                end_time = omp_get_wtime();     
                cout << "Parallel Bubble Sort time = " << end_time - start_time << " seconds.\n";
                break;
            case 2:
                cout << "Merge Sort\n";
                // Sequential time
                start_time = omp_get_wtime();
                mergeSort(crr, 0, n - 1);
                end_time = omp_get_wtime();     
                cout << "Sequential Merge Sort time = " << end_time - start_time << " seconds.\n";
                
                // Parallel time
                start_time = omp_get_wtime();
                parallelMergeSort(drr, 0, n - 1);
                end_time = omp_get_wtime();     
                cout << "Parallel Merge Sort time = " << end_time - start_time << " seconds.\n";
                break;
            case 3:
                flag = false;
                break;
            default:
                cout<<"Invalid choice\n";
                break;
        }
    }
    return 0;
    
}   

/*
To run this
1. compile using : g++ -fopenmp assg2.cpp
2. run using     : ./a.out
*/