#ifndef NEURAL_H_
#define NEURAL_H_

#include<iostream>
#include<stdio.h>
#include<vector>
#include<math.h>
#include <climits>
#include <random>
#include "Matrix.h"

namespace Neural{
    class layor{
        private:
            Matrix z,a;
            Matrix w;
            Matrix b;
            Matrix d;
            Matrix In;
            int l;
            double alpha;
        public:
            layor(int _l): z(_l,1),a(_l,1),b(_l,1),d(_l,1),w(0,0),In(0,0){
                l = _l;
                for(int i = 0; i < _l; i++){
                    z[i][0] = 0;
                    a[i][0] = 0;
                }
            }
            int node_cn(){return l;}
            void set_W(int l1){
                Matrix m(l,l1);
                m.set_random();
                w = m;
                Matrix n(l,1);
                n.set_random();
                b = n;
            }
            Matrix fw(const Matrix F){
                In = F;
                z = w*F+b;
                for(int i = 0; i < l; i++){
                    a[i][0] = f(z[i][0]);
                }
                return a;
            }
            Matrix bc(const Matrix D,const Matrix W){
                for(int i = 0; i < l; i++){
                    double sum = 0;
                    for(int k = 0; k < D.c; k++){
                        sum += D[k][0]*W[k][i];
                    }
                    d[i][0] = sum;
                }
                for(int i = 0; i < l; i++){
                    d[i][0] *= df(z[i][0]);
                }
                return d;
            }
            void reset_W(){
                for(int i = 0; i < l; i++){
                    for(int j = 0; j < In.c; j++){
                        w[i][j] -= alpha*d[i][0]*In[j][0];
                    }     
                    b[i][0] -= alpha*d[i][0];
                }
            }
            virtual double f(double x);
            virtual double df(double x);
    };
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