#ifndef LAYOR_H_
#define LAYOR_H_

#include<iostream>
#include<vector>
#include "Matrix.h"

namespace Neural{
    double sigmoid(double x){
        return 1/(1+std::exp(-x));
    }
    double dsigmoid(double x){
        return sigmoid(x)*(1-sigmoid(x));
    }
    double tanh(double x){
        return std::tanh(x);
    }
    double dtanh(double x){
        return 1/std::cosh(x);
    }
    double identity(double x){
        return x;
    }
    double didentity(double x){
        return 1;
    }
    double relu(double x){
        return std::max(x,(double)0);
    }
    double drelu(double x){
        return (x > 0)? 1 : 0;
    }
    double likely_relu(double x){
        return std::max(x,0.01*x);
    }
    double dlikely_relu(double x){
        return (x > 0)? 1 : 0.01;
    }
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
            layor(int _l,double _alpha): z(_l,1),a(_l,1),b(_l,1),d(_l,1),w(0,0),In(0,0){
                l = _l;
                alpha = _alpha;
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
            Matrix out_bc(const Matrix ANS){
                d = a-ANS;
                for(int i = 0; i < l; i++){
                    d[i][0] *= df(z[i][0]);
                }
                return d;
            }
            Matrix w_push(){
                Matrix W = w;
                return W;
            }
            void reset_W(){
                for(int i = 0; i < l; i++){
                    for(int j = 0; j < In.c; j++){
                        w[i][j] -= alpha*d[i][0]*In[j][0];
                    }     
                    b[i][0] -= alpha*d[i][0];
                }
            }
            virtual double f(double x) = 0;
            virtual double df(double x) = 0;
    };
    class sigmoid_layor : public layor
    {
        public:
            sigmoid_layor(int _l,double _alpha): layor(_l,_alpha){}
            double f(double x){
                return sigmoid(x);
            }
            double df(double x){
                return dsigmoid(x);
            }
    };
    class tanh_layor : public layor{
        public:
            tanh_layor(int _l,double _alpha): layor(_l,_alpha){}
            double f(double x){
                return tanh(x);
            }
            double df(double x){
                return dtanh(x);
            }
    };
    class identity_layor : public layor{
        public:
            identity_layor(int _l,double _alpha): layor(_l,_alpha){}
            double f(double x){
                return identity(x);
            }
            double df(double x){
                return didentity(x);
            }
    };
    class relu_layor : public layor{
        public:
            relu_layor(int _l,double _alpha): layor(_l,_alpha){}
            double f(double x){
                return relu(x);
            }
            double df(double x){
                return drelu(x);
            }
    };
    class likely_relu_layor : public layor{
        public:
            likely_relu_layor(int _l,double _alpha): layor(_l,_alpha){}
            double f(double x){
                return likely_relu(x);
            }
            double df(double x){
                return dlikely_relu(x);
            }
    };
}

#endif // LAYOR_H_