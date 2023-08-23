
    std::vector<Neural::Matrix> M(In);
    std::vector<double> A = {1,0,0};
    Neural::Matrix Ans(3,1);
    for(int i = 0; i < 3; i++){
        Ans[i][0] = A[i];
    }
    for(int i = 0; i < 4000; i++){
        std::cout <<"------------"<<std::endl;
        if(i % 100 == 0)C.keisan(M).show();
        else C.keisan(M);
        C.fix(A);
        std::cout <<"------------"<<std::endl;   
    }