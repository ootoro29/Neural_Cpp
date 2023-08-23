#ifndef CNN_H_
#define CNN_H_

#include<iostream>
#include<stdio.h>
#include<vector>
#include <climits>
#include <random>
#include "Matrix.h"
#include "layor.h"
#include "Network.h"

namespace Neural{
    class CNN{
        private:
            std::vector<std::vector<Matrix>> w_F;
            Matrix b_F;
            std::vector<std::vector<Matrix>> w_O;
            Matrix b_O;
            std::vector<Matrix> In;
            Matrix Out;
            std::vector<Matrix>d_F;
            Matrix d_O;
            Matrix z_o;
            Matrix a_o;
            std::vector<Matrix>z_f;
            std::vector<Matrix>a_f;
            std::vector<Matrix>a_p;
            int p_slide = 2;
            int p;
            int l2;
            int k_channel;
            int c;
            int a_len;
            int out_size = 0;
            double alpha = 0.001;//0.0001
            double alpha2 = 0.002;//0.0002
            std::vector<std::vector<std::pair<int,int>>> max_index;
        public:
            CNN(int _l,int _a,int _k,int _c,int _o):In(_k,Matrix(_l,_l)),b_F(_k,1),b_O(_o,1),Out(_o,1),d_O(_o,1),z_o(_o,1),a_o(_o,1){
                //_l:入力サイズ,_a:畳み込みサイズ,_k:畳み込みチャンネルサイズ ,_c:入力チャンネルサイズ
                out_size = _o;
                k_channel = _k;
                c = _c;
                a_len = _a;
                l2 = _l -_a + 1;
                assert(l2%p_slide == 0);
                p = l2/p_slide;
                max_index.resize(p);
                for(int i = 0; i < p; i++){
                    for(int j = 0; j < p; j++){
                        max_index[i].push_back(std::make_pair(0,0));
                    }
                }
                w_O = std::vector(_o,std::vector(0,Matrix(0,0)));
                w_F = std::vector(_c,std::vector(0,Matrix(0,0)));
                d_F = std::vector<Matrix>(k_channel,Matrix(l2,l2));
                z_f = std::vector<Matrix>(k_channel,Matrix(l2,l2));
                a_f = std::vector<Matrix>(k_channel,Matrix(l2,l2));
                a_p = std::vector<Matrix>(k_channel,Matrix(p,p));
                std::vector<std::vector<Matrix>> m(_c,std::vector(_k,Matrix(a_len,a_len)));
                std::vector<std::vector<Matrix>> m2(_o,std::vector(_k,Matrix(p,p)));
                for(int i = 0; i < _o; i++){
                    for(int j = 0; j < _k; j++){
                        m2[i][j].set_random();
                        w_O[i].push_back(m2[i][j]);
                    }
                }
                for(int i = 0; i < _c; i++){
                    for(int j = 0; j < _k; j++){
                        m[i][j].set_random();
                        w_F[i].push_back(m[i][j]);   
                    }
                }
                Matrix n(_k,1);
                Matrix n2(_o,1);
                n.set_random();
                n2.set_random();
                b_F = n;
                b_O = n2;
            }
        Matrix keisan(std::vector<Matrix> IX){
            assert(IX.size() == c);
            for(int i = 0; i < c; i++){
                In[i] = IX[i];
            }
            
            for(int k = 0; k < k_channel; k++){
                for(int i = 0; i < l2; i++){
                    for(int j = 0; j < l2; j++){
                        double x = 0;
                        for(int ay = 0; ay < a_len; ay++){
                            for(int ax = 0; ax < a_len; ax++){
                                for(int ic = 0; ic < c; ic++){
                                    x += In[ic][i+ay][j+ax]*w_F[ic][k][ay][ax];
                                }
                            }    
                        }
                        z_f[k][i][j] = x;
                    }    
                }
                /*
                std::cout <<k<<"/"<<k_channel<<"-----------------------" <<std::endl;
                z_f[k].show();
                std::cout <<"--------------------------" <<std::endl;
                */
            }
            for(int k = 0; k < k_channel; k++){
                for(int i = 0; i < l2; i++){
                    for(int j = 0; j < l2; j++){
                        a_f[k][i][j] = f_F(z_f[k][i][j]);
                    }    
                }
            }
            
            for(int k = 0; k < k_channel; k++){
                for(int i = 0; i < p; i++){
                    for(int j = 0; j < p; j++){
                        double max = a_f[k][i*p_slide][j*p_slide];
                        std::pair<int,int>index;
                        for(int i2 = 0; i2 < 1; i2++){
                            for(int j2 = 0; j2 < 1; j2++){
                                if(max <= a_f[k][i*p_slide+i2][j*p_slide+j2]){
                                    max = a_f[k][i*p_slide+i2][j*p_slide+j2];
                                    index.first = i*p_slide+i2;
                                    index.second = j*p_slide+j2;
                                }
                            }    
                        }
                        a_p[k][i][j] = max;
                        max_index[i][j] = index;       
                    }    
                }   
            }
            
            for(int ind = 0; ind < out_size; ind++){
                double x = 0;
                for(int k = 0; k < k_channel; k++){
                    for(int i = 0; i < p; i++){
                        for(int j = 0; j < p; j++){
                            x += w_O[ind][k][i][j]*a_p[k][i][j];
                        }    
                    }   
                }
                z_o[ind][0] = x;
            }
            for(int i = 0; i < out_size; i++){
                a_o[i][0] = f_O(z_o[i][0]);
            }
            Out = a_o;
            return Out;
        }
        void fix(Matrix ANS){
            for(int i = 0; i < out_size; i++){
                d_O[i][0] = (Out[i][0] - ANS[i][0])*df_O(z_o[i][0]);
            }
            for(int k = 0; k < k_channel; k++){
                for(int i = 0; i < l2; i++){
                    for(int j = 0; j < l2; j++){
                        double x = 0;
                        int i_ = i / p_slide;
                        int j_ = j / p_slide;
                        for(int o = 0; o < out_size; o++){
                            x += d_O[o][0]*w_O[o][k][i_][j_];
                        }
                        x *= df_F(z_f[k][i][j]);
                        std::pair<int,int>P = max_index[i_][j_];
                        if(P.first != i || P.second != j){
                            x *= 0;
                        }
                        d_F[k][i][j] = x;
                    }
                }
            }
            for(int o = 0; o < out_size; o++){
                for(int k = 0; k < k_channel; k++){
                    for(int i = 0; i < p; i++){
                        for(int j = 0; j < p; j++){
                            w_O[o][k][i][j] -= alpha2*d_O[o][0]*a_p[k][i][j];
                        }    
                    }
                }
                b_O[o][0] -= alpha2*d_O[o][0];
            }
            for(int k = 0; k < k_channel; k++){
                double x = 0;
                for(int i = 0; i < a_len; i++){
                    for(int j = 0; j < a_len; j++){
                        x += d_F[k][i][j];
                    }
                }
                b_F[k][0] -= alpha*x;
            }
            for(int k = 0; k < k_channel; k++){
                for(int i = 0; i < l2; i++){
                    for(int j = 0; j < l2; j++){
                        for(int ic = 0; ic < c; ic++){
                            double x = 0;
                            for(int ay = 0; ay < a_len; ay++){
                                for(int ax = 0; ax < a_len; ax++){
                                    x += In[ic][i+ay][j+ax] * d_F[k][ay][ax];
                                    
                                }    
                            }
                            w_F[ic][k][i][j] -= alpha * x;
                        }
                    }    
                }
            }
        }
        private:
            double f_F(double x){
                return tanh(x);
            }
            double df_F(double x){
                return dtanh(x);
            }
            double f_O(double x){
                return identity(x);
            }
            double df_O(double x){
                return didentity(x);
            }
    };
}

#endif // CNN_H_