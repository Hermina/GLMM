function [precision, recall, f, NMI_score, num_of_edges] = graph_learning_perf_eval(L_0,L)
% evaluate the performance of graph learning algorithms

%% edges in the groundtruth graph
L_0tmp = L_0-diag(diag(L_0));
edges_groundtruth = squareform(L_0tmp)~=0;

%% edges in the learned graph
Ltmp = L-diag(diag(L));
edges_learned = squareform(Ltmp)~=0;

%% recall & precision
[R,P] = perfcurve(double(edges_groundtruth),double(edges_learned),1,'xCrit','reca','yCrit','prec');
precision = P(2);
recall = R(2);

%% F-measure
f = 2*precision*recall/(precision+recall);
if isnan(f)
    f = 0;
end

%% NMI
NMI_score = nmi(double(edges_learned)',double(edges_groundtruth)');
if isnan(NMI_score)
    NMI_score = 0;
end

%% number of edges in the learned graph
num_of_edges = sum(edges_learned);