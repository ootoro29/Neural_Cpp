#include<iostream>
#include<vector>
#include "../inc/Matrix.h"
#include "../inc/layor.h"
#include "../inc/Network.h"

int main(){
    std::vector<Neural::layor*>L;
    L.push_back(new Neural::relu_layor(3,1.5));
    L.push_back(new Neural::sigmoid_layor(7,0.2));
    L.push_back(new Neural::identity_layor(2,0.4));
    Neural::NetWork N_N(L);
    std::vector<std::vector<double>>Data_S = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};
    std::vector<std::vector<double>>Data_A = {{0,1},{0,1},{1,0},{1,1},{0,0},{1,0},{0,1},{1,1}};
    long int step = 0;
    while(1){
        printf("STEP:%ld\n",step);
        for(int i = 0; i < 10; i++){
            for(int j = 0; j < Data_A.size(); j++){
                
                step++;
                N_N.learning(Data_S[j],Data_A[j]);
                if(i == 9){
                    N_N.dis_out();
                }
            }
        }
        int c;
        scanf("%c",&c);
    }
    return 0;
}
