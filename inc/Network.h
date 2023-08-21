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
            NetWork(std::vector<layor>_L): out(_L[_L.size()-1].node_cn(),1),ans(_L[_L.size()-1].node_cn(),1),in(_L[0].node_cn(),1){
                N = _L.size();
                for(int i = 0; i < N; i++){
                    net.push_back(_L[i]);
                }
                for(int i = 1; i < N; i++){
                    net[i].set_W(_L[i-1].node_cn());
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