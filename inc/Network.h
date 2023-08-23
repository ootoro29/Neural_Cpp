#ifndef NEURAL_H_
#define NEURAL_H_

#include<iostream>
#include<stdio.h>
#include<vector>
#include <climits>
#include <random>
#include "Matrix.h"
#include "layor.h"

namespace Neural{
    class NetWork{
        private:
            int N;
            double eta = 1;
            std::vector<Neural::layor*>net;
            Matrix ans;
            Matrix in;
            Matrix out;
        
        public:
            NetWork(std::vector<layor*>_L): out(_L[_L.size()-1]->node_cn(),1),ans(_L[_L.size()-1]->node_cn(),1),in(_L[0]->node_cn(),1){
                N = _L.size();
                for(int i = 0; i < N; i++){
                    net.push_back(_L[i]);
                }
                for(int i = 1; i < N; i++){
                    net[i]->set_W(_L[i-1]->node_cn());
                }
            }
            void learning(std::vector<double> inx,std::vector<double> aa){
                for(int i = 0; i < net[0]->node_cn(); i++){
                    in[i][0] = inx[i];
                }
                for(int i = 0; i < net[N-1]->node_cn(); i++){
                    ans[i][0] = aa[i];
                }
                keisan(inx);
                fix();
            }
            void dis_out(){
                for(int i = 0; i < in.c; i++){
                    std::cout << in[i][0] << " ";
                }
                std::cout <<":-> ";
                for(int i = 0; i < out.c; i++){
                    std::cout << out[i][0] << " ";
                }
                std::cout <<std::endl;
            }
        private:
            void keisan(std::vector<double> inx){
                Matrix X(inx);
                for(int i = 1; i < N; i++){
                    X = net[i]->fw(X);
                }
                for(int i = 0; i < X.c; i++){
                    out[i][0] = X[i][0];
                }
            }
            void fix(){
                Matrix C = net[N-1]->out_bc(ans);
                Matrix W = net[N-1]->w_push();
                for(int i = 0; i < N-1; i++){
                    int index = N-i-2;
                    C = net[index]->bc(C,W);
                    W = net[index]->w_push();
                }
                for(int i = 1; i < N; i++){
                    net[i]->reset_W();
                }
            }
    };
}

#endif // NEURAL_H_