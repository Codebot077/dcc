
#include<bits/stdc++.h>
#include<omp.h>

using namespace std;
// Function to find the minimum value in the array
int minval(int arr[], int n){
  int minval = arr[0];
  #pragma omp parallel for reduction(min : minval)
    for(int i = 0; i < n; i++){
      if(arr[i] < minval) minval = arr[i];
    }
  return minval;
}
// Function to find the maximum value in the array
int maxval(int arr[], int n){
  int maxval = arr[0];
  #pragma omp parallel for reduction(max : maxval)
    for(int i = 0; i < n; i++){
      if(arr[i] > maxval) maxval = arr[i];
    }
  return maxval;
}
// Function to find the sum of the elements in the array
int sum(int arr[], int n){
  int sum = 0;
  #pragma omp parallel for reduction(+ : sum)
    for(int i = 0; i < n; i++){
      sum += arr[i];
    }
  return sum;
}
// Function to find the average of the elements in the array
int average(int arr[], int n){
  return (double)sum(arr, n) / n;
}

int main(){
  
  int n;
  cout<<"Enter the number of elements: ";
  cin>>n;
  int arr[n];
  cout<<"Enter the elements: ";
  for(int i=0;i<n;i++){
    arr[i] = 1 + (rand() % n);
  }
  bool flag = true;
  while(flag){
     cout << "-------------------------- Menu -------------------------------"  << endl;
      cout << "1. Sequential \n";
      cout << "2. Parallel\n";
      cout<< "3. Exit\n";
      cout << "---------------------------------------------------------------"  << endl;
      int choice =-1;
      cout<<"Enter your choice: ";
      cin>>choice;
      double start_time = omp_get_wtime();
      switch(choice){
        case 1:
          {cout<<"Sequential\n";
          cout << "The minimum value is: " << *min_element(arr,arr+ n) << '\n';
          cout<< "The maximum value is: "<< *max_element(arr,arr+n) << '\n';
          int sum1=0;
          for(int i=0;i<n;i++){
            sum1+=arr[i];
          }
          cout<<"The summation is: "<<sum1<<'\n';
          cout<<"The average is: "<<(double)sum1/(double)n<<'\n';
          break;}
        case 2:
          {cout<<"Parallel\n";
          cout<<"The minimum value is: "<<minval(arr,n)<<'\n';
          cout << "The maximum value is: " << maxval(arr, n) << '\n';
          cout << "The summation is: " << sum(arr, n) << '\n';
          cout << "The average is: " << average(arr, n) << '\n';
          break;}
        case 3:
          flag = false;
          break;
        default:
          cout << "Invalid choice\n";
      }
      double end_time = omp_get_wtime();
      cout<<"Time taken: "<<end_time-start_time<<" seconds\n";
  }
  return 0;
}

/*
To run this
1. compile using : g++ -fopenmp assg3.cpp
2. run using     : ./a.out
*/