function [ U,obj ] = jz_ADMM( X,L,step,num_iter,gamma,lambda)

[M,N] = size(X);

% Initialize
r1 = step; 
r2 = step;
U = randn(M,N);
S = randn(M,N);
H = randn(M,N);
A1 = zeros(M,N);
A2 = zeros(M,N);

obj = zeros(num_iter,1); % Watch objective change
% obj_aug = zeros(num_iter*4,1);

tic
for i = 1:num_iter
    i
    
    % Update U
    Z1 = X-S+A1/r1; Z2 = H+A2/r2;
    rz = (r1*Z1 + r2*Z2)/(r1+r2);
    [P,Sig,D] = svd(rz,'econ');
    U = P*jz_shrink(Sig,1/(r1+r2))*D';
    
    [Uu,Ss,Vv] = svd(U,'econ');
%     obj_aug(4*i-3) = trace(Ss) + gamma*norm(S(:),1) + lambda*trace(H*L*H') ...
%         + trace((X-U-S)'*A1) + r1/2*norm(X-U-S,'fro')^2 + trace((H-U)'*A2) + r2/2*norm(H-U,'fro')^2;
    
    % Update S
    S = jz_shrink(X-U+A1/r1,gamma/r1);
    
%     obj_aug(4*i-2) = trace(Ss) + gamma*norm(S(:),1) + lambda*trace(H*L*H') ...
%         + trace((X-U-S)'*A1) + r1/2*norm(X-U-S,'fro')^2 + trace((H-U)'*A2) + r2/2*norm(H-U,'fro')^2;
    
    % Update H
    H = r2*(U-A2/r2)/(2*lambda*L + r2*eye(N));
    
%     obj_aug(4*i-1) = trace(Ss) + gamma*norm(S(:),1) + lambda*trace(H*L*H') ...
%         + trace((X-U-S)'*A1) + r1/2*norm(X-U-S,'fro')^2 + trace((H-U)'*A2) + r2/2*norm(H-U,'fro')^2;
    
    % Update A1, A2
    A1 = A1 + r1*(X-U-S);
    A2 = A2 + r2*(H-U);
    
%     obj_aug(4*i-0) = trace(Ss) + gamma*norm(S(:),1) + lambda*trace(H*L*H') ...
%         + trace((X-U-S)'*A1) + r1/2*norm(X-U-S,'fro')^2 + trace((H-U)'*A2) + r2/2*norm(H-U,'fro')^2;
    
    % Save objective value
     obj(i) = trace(Ss) + gamma*norm(S(:),1) + lambda*trace(H*L*H');

toc    
end

disp(norm(H - U)/norm(U))
disp(norm(X-S-U)/norm(X))
% plot(obj);
end

