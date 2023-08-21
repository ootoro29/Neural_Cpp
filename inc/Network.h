#ifndef NEURAL_H_
#define NEURAL_H_

#include<iostream>
#include<stdio.h>
#include<vector>
#include<math.h>
#include <climits>
#include <random>
#include "Matrix.h"
#include "layor.h"

namespace Neural{
    class NetWork{
        private:
            int N;
            double eta = 1;
            std::vector<layor>net;
            Matrix ans;
            Matrix in;
            Matrix out;
        
        public:
            NetWork(std::vector<int>k): out(k[k.size()-1],1),ans(k[k.size()-1],1),in(k[0],1){
                N = k.size();
                for(int i = 0; i < N; i++){
                    layor L(k[i]);
                    net.push_back(L);
                }
                for(int i = 1; i < N; i++){
                    net[i].set_W(k[i-1]);
                }
            }
            void learning(std::vector<double> inx,std::vector<double> aa);
            void dis_out();
        private:
            void keisan(std::vector<double> inx);
            void fix(Matrix in);
        
    };
}

#endif // NEURAL_H_