clear all
close all
%% generate graphs
n = 15; %graph size
m = 150; %number of signals
k = 2 ; %number of clusters
zero_thresh = 10e-4;

g(k) = gsp_erdos_renyi(n,0.7);
for i = 1:k
    while(1)
    	g(i) = gsp_erdos_renyi(n, 0.7);
        eigs = sort(eig(g(i).L));
        if (eigs(2) > zero_thresh) %ensuring graphs are connected
            break;
        end
    end
end
gamma = rand([m,1]);
gamma_cut = zeros(m,k);
dist = 0.5;
%p = [0, 0.2, 0.6, 1];
p = 0:1/k:1;
y = zeros(m,n);
true_y = zeros(m,n,k);
center = zeros(n,k);
gauss = zeros(n, n, k);
Lap = zeros(n, n, k);
for i=1:k
    gc = pinv(full(g(i).L));
    gauss(:,:,i) = (gc +gc')/2;
    Lap(:,:,i) = full(g(i).L);
    %% generate centers, responsibilities and data
    center(:,i) = dist * randn([n,1]);
    center(:,i) = center(:,i) - mean(center(:,i));
    gamma_cut(p(i)<gamma & gamma<=p(i+1), i) = 1;
    true_y(:,:,i) = squeeze(gamma_cut(:,i)).*mvnrnd(center(:,i),gauss(:,:,i),m);
    y = y + true_y(:,:,i);
end


%% train a glmm on data y
iterations = 200;
[Ls, gamma_hats, mus] = glmm_matlab(y, iterations,k);
disp('Training done')

%%

disp(sum(gamma_hats,1));

[identify, precision, recall,  f, cl_errors] = identify_and_compare(Ls, Lap, gamma_hats, gamma_cut, k)


