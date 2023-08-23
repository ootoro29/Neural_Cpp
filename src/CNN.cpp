#include<iostream>
#include<vector>
#include "../inc/Matrix.h"
#include "../inc/layor.h"
#include "../inc/Network.h"
#include "../inc/CNN.h"

int main(){
    Neural::CNN C(7,4,36,2,3);
    double in[7][7] = {0,1,1,0,1,0,0,
                       0,0,1,1,1,0,1,
                       1,0,0,0,1,0,1,
                       1,1,1,0,1,0,0,
                       0,1,1,1,1,0,0,
                       1,1,1,1,0,1,0,
                       0,0,1,1,1,0,0};
    double in2[7][7] = {0,1,0,0,1,0,0,
                       1,0,0,1,1,0,1,
                       0,1,0,0,1,0,1,
                       1,1,1,0,1,0,0,
                       1,0,0,1,1,0,0,
                       1,0,1,1,0,1,0,
                       1,0,1,1,1,0,0};
    std::vector<std::vector<double>>I(7,std::vector<double>(7));
    for(int i = 0; i < 7; i++){
        for(int j = 0; j < 7; j++){
            I[i][j] = in[i][j];
        }
    }
    std::vector<std::vector<double>>I2(7,std::vector<double>(7));
    for(int i = 0; i < 7; i++){
        for(int j = 0; j < 7; j++){
            I2[i][j] = in2[i][j];
        }
    }
    std::vector<Neural::Matrix> In;
    In.push_back(I);
    In.push_back(I2);
    std::vector<Neural::Matrix> M(In);
    std::vector<double> A = {3,-2,1};
    Neural::Matrix Ans(3,1);
    for(int i = 0; i < 3; i++){
        Ans[i][0] = A[i];
    }
    for(int i = 0; i < 200; i++){
        if(i%10 == 0){
            std::cout<<"---------------"<<std::endl;
            C.keisan(M).show();
            std::cout<<"---------------"<<std::endl;
        }else{
            C.keisan(M);
        }
        C.fix(A);
    }
    return 0;
}
