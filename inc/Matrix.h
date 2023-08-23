#ifndef MATRIX_H_
#define MATRIX_H_
#include <assert.h>
#include<iostream>
#include<vector>
#include <random>
namespace Neural{
    typedef std::vector<std::vector<double>> v2;    
    class Matrix{
        void error(){
            printf("ERROR");
            exit(0);
        }
        public:
            int c = 0,r = 0;
            v2 arr;
            Matrix(int c,int r){
                this->c = c;
                this->r = r;
                arr.resize(c);
                for(int i = 0; i < c; i++){
                    for(int j = 0; j < r; j++){
                    arr[i].push_back(0);
                    }   
                }
            }
            Matrix(v2 x){
                c = x.size();
                for(int i = 0;i < c; i++){
                    r = std::max(r,(int)x[i].size());
                }
                arr.resize(c);
                for(int i = 0; i < c; i++){
                    for(int j = 0; j < r; j++){
                        if(j < x[i].size()){
                            arr[i].push_back(x[i][j]);
                        }else{
                            arr[i].push_back(0);
                        }
                    }
                }
            }
            Matrix(std::vector<double> x){
                c = x.size();
                r = 1;
                arr.resize(c);
                for(int i = 0; i < c; i++){
                    arr[i].push_back(x[i]);
                }
            }
            void set(v2 x){
                for(int i = 0; i < c; i++){
                    for(int j = 0; j < r; j++){
                    int p = x[i][j];
                    arr[i].push_back(p);
                    }   
                }
            }
            void k_line(double k,int index){
                if(index >= c)return;
                for(int j = 0; j < r; j++){
                    arr[index][j] *= k;
                }
            }
            void add_line(int index,double k,int index2){
                if(index >= c || index2 >= c)return;
                for(int j = 0; j < r; j++){
                    arr[index][j] += arr[index2][j]*k;
                }
            }
            void change_line(int index1,int index2){
                if(index1 >= c || index2 >= c)return;
                std::vector<double>l_m;
                for(int j = 0; j < r; j++){
                    l_m.push_back(arr[index2][j]);
                    arr[index2][j] = arr[index1][j];
                }
                for(int j = 0; j < r; j++){
                    arr[index1][j] = l_m[j];
                }
                
            }
            void show(){
                for(int i = 0; i < c; i++){
                    for(int j = 0; j < r; j++){
                    std::cout << arr[i][j] << " ";
                    }   
                    std::cout << std::endl;
                }
            }
            void set_random(){
                std::random_device rnd;
                for(int i = 0; i < c; i++){
                    for(int j = 0; j < r; j++){
                        arr[i][j] = (abs((int)rnd())%10 - 5)/2.5;
                    }
                }    
            }
            Matrix cofactor(int in_c,int in_r){
                Matrix ans(c-1,r-1);
                int cn_c = 0,cn_r = 0;
                for(int i = 0; i < c; i++){
                    if(i != in_c){
                        for(int j = 0; j < r; j++){
                            if(j != in_r){
                                ans[cn_c][cn_r] = arr[i][j];
                                cn_r++;       
                            }    
                        }
                        cn_c++;
                    }   
                }
                return ans;
            }
            double det(){
                if(c != r){
                    error();
                }else{
                    if(c == 2){
                        return arr[0][0]*arr[1][1] - arr[0][1]*arr[1][0]; 
                    }else{
                        double ans = 0;
                        for(int i = 0; i < c; i++){
                            ans += arr[i][0]*(cofactor(i,0).det());
                        }
                    }
                }
            }
            Matrix operator +(const Matrix &m){
                if(!(c == m.c && r == m.r))error();
                Matrix ans(c,r);
                for(int i = 0; i < c; i++){
                    for(int j = 0; j < r; j++){
                        ans[i][j] = arr[i][j] + m[i][j];
                    }    
                }
                return ans;
            }
            Matrix operator -(const Matrix &m){
                if(!(c == m.c && r == m.r))error();
                Matrix ans(c,r);
                for(int i = 0; i < c; i++){
                    for(int j = 0; j < r; j++){
                        ans[i][j] = arr[i][j] - m[i][j];
                    }    
                }
                return ans;
            }
            Matrix operator *(const Matrix &m){
                if(!(r == m.c))error();
                Matrix ans(c,m.r);
                for(int i = 0; i < c; i++){
                    for(int j = 0; j < m.r; j++){
                        double x = 0;
                        for(int k = 0; k < r; k++){
                            x += arr[i][k]*m[k][j];
                        }
                        ans[i][j] = x;
                    }    
                }
                return ans;
            }
            Matrix operator *(double k){
                Matrix ans(c,r);
                for(int i = 0; i < c; i++){
                    for(int j = 0; j < r; j++){
                        ans[i][j] = arr[i][j]*k;
                    }    
                }
                return ans;
            }
            
            Matrix operator !(){
                if(c!=r)error();
                Matrix t(arr);
                Matrix ans(c,r);
                bool d = false;
                for(int i = 0; i < c; i++){
                    for(int j = 0; j < r; j++){
                        if(i == j){
                            ans[i][j] = 1;
                        }else{
                            ans[i][j] = 0;
                        }
                    }
                }
                for(int i = 0; i < c; i++){
                    if(t[i][i] == 0){
                        int inx = 0;
                        for(int j = i+1; j < c; j++){
                            inx = j;
                            if(t[j][i] != 0){
                                break;
                            }
                            if(j == c-1)inx = -1;
                        }
                        if(inx == -1){
                            d = true;
                            break;
                        }
                        t.change_line(i,inx);
                        ans.change_line(i,inx);
                    }else{
                        double k1 = 1.0/t[i][i];
                        t.k_line(k1,i);
                        ans.k_line(k1,i);
                        for(int j = 0; j < c; j++){
                            if(i!=j){
                                double k2 = -t[j][i];
                                t.add_line(j,k2,i);
                                ans.add_line(j,k2,i);
                            }
                        }
                    }
                }
                if(d){
                    printf("Can't inverse\n");
                    Matrix err(c,r);
                    return err;
                }
                return ans;
            }
            Matrix operator &(){
                Matrix ans(r,c);
                for(int i = 0; i < c; i++){
                    for(int j = 0; j < r; j ++){
                        ans[j][i] = arr[i][j];
                    }
                }
                return ans;
            }
            Matrix &operator =(Matrix m){
                r = m.r;
                c = m.c;
                arr.erase(arr.begin(),arr.end());
                arr.resize(c);
                for(int i = 0; i < c; i++){
                    for(int j = 0; j < r; j++){
                        arr[i].push_back(m[i][j]);
                    }
                }
                return *this;
            }
            std::vector<double> const operator [](int index) const{
                return arr[index];
            }
            double* operator [](int index){
                return &(*arr[index].begin());
            }
    };
}


#endif // MATRIX_H_