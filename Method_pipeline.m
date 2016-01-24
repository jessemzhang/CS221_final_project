
for lab = {'Buettner_keras_tanh_120hidden1_40hidden2_30hidden3_5_20_recon'}

type = lab{1}

% Load low-dimensional matrix
Xk = dlmread(['Xk_' type '.txt']);
[M,N] = size(Xk);

% Generate similarity network
% A = abs(corrcov(cov(X))-diag(ones(N,1)));
sig = 0.8;
A = (exp(-squareform(pdist(Xk')).^2/(2*sig^2))); %-diag(ones(N,1)));
D = diag(sum(A));
L = D-A;

% Load noisy data matrix
lab_name = strsplit(type,'_');
X = dlmread(['./data/' lab_name{1} '/' lab_name{1} '_expression.txt']);

% Solve optimization problem
step = 2;
num_iter = 30;
gamma = 1;

disp(size(Xk))
disp(size(X))
disp(size(L))

for lambda = [10000,100000,1000000,0.00001,0.0001,0.001,0.01,0.1,1,10,50,100,500,1000]

[U,obj] = jz_ADMM(X,L,step,num_iter,gamma,lambda);

dlmwrite(['./ADMM_results/U_' type '_' num2str(lambda) '.txt'],U,'\t')
dlmwrite(['./ADMM_results/obj_' type '_' num2str(lambda) '.txt'],obj,'\t')

end

end
