#ifndef LAYOR_H_
#define LAYOR_H_

#include<iostream>
#include<vector>
#include "Matrix.h"

namespace Neural{
    class sigmoid_layor : layor{
        double f(double x){
            return 1/(1+std::exp(-x));
        }
        double df(double x){
            return f(x)*(1-f(x));
        }
    };
    class tanh_layor : layor{
        double f(double x){
            return std::tanh(x);
        }
        double df(double x){
            return 1/std::cosh(x);
        }
    };
    class identity_layor : layor{
        double f(double x){
            return x;
        }
        double df(double x){
            return 1;
        }
    };
    class relu_layor : layor{
        double f(double x){
            return std::min(x,(double)0);
        }
        double df(double x){
            return (x > 0)? 1 : 0;
        }
    };
    class likely_relu_layor : layor{
        double f(double x){
            return std::min(x,0.01*x);
        }
        double df(double x){
            return (x > 0)? 1 : 0.01;
        }
    };
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
}

#endif // LAYOR_H_