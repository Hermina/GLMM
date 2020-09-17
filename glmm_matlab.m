function [L, gamma_hat, mu, log_likelihood] = glmm_matlab(y, iterations, classes, spread, regul, norm_par)

if (nargin == 3)
    spread = 0.1;
    regul = 0.15;
    norm_par = 1.5;%0.3;%1.2;
end

delta = 2;
n = size(y,2);
m = size(y,1); 
%% initialise
L = zeros(n,n,classes);
W = zeros(n,n,classes);
sigma = zeros(n-1,n-1,classes);
mu = zeros(n, classes);
gamma_hat = zeros(m, classes);
p = zeros(classes,1);
vecl = zeros(n,n,classes);
vall = zeros(n,n,classes);
yl = zeros(m, n-1, classes);
for class = 1:classes
    L(:,:,class) = spread*eye(n) - spread/n *ones(n);
    mu_curr = mean(y,1) + randn(1,n).* std(y,1); 
    mu(:,class) = mu_curr - mean(mu_curr);
    p(class) = 1/classes;
end

%% start the algorithm
for it = 1:iterations
    %Expectation step
    %putting everything in eigenvector space of dim-1
    pall = 0;
    for class = 1:classes
        [vecl(:,:,class), vall(:,:,class)] = eig(squeeze(L(:,:,class)));
        sigma(:,:,class) = inv(vall(2:n,2:n,class) + regul*eye(n-1)); %constraining covariance in all directions
        sigma(:,:,class) = (squeeze(sigma(:,:,class)) + squeeze(sigma(:,:,class))')/2;
        yl(:,:,class) = (y-mu(:,class)')*vecl(:,2:n,class); 
        pall = pall + p(class) * mvnpdf(yl(:,:,class), zeros(1,n-1), sigma(:,:,class));
    end
    %compute cluster probabilities gamma_hat  
    pall(pall == 0) = 0.1;
    for class = 1:classes
        gamma_hat(:,class) = (p(class) * mvnpdf(yl(:,:,class), zeros(1,n-1), sigma(:,:,class)))./pall;
    end
    log_likelihood(it) = sum(log(pall));

    %Maximisation step: update mu, W and p
    for class = 1:classes
        mu(:,class) = (gamma_hat(:,class)'*y)/sum(gamma_hat(:,class));
        yc = repmat(sqrt(gamma_hat(:,class)),[1,n]) .* (y - mu(:,class)'); 
        Z = gsp_distanz(yc).^2;
        theta = mean(Z(:))/norm_par;
        W_curr = delta*gsp_learn_graph_log_degrees(Z ./ theta, 1, 1);
        W(:,:,class) = W_curr;
        p(class) = sum(gamma_hat(:,class))/m;
        %compute Ls
        L(:,:,class) = diag(sum(W(:,:,class),2)) - W(:,:,class);
        W_curr(W_curr<1e-3) = 0;
        W(:,:,class) = W_curr;
    end
end
end
