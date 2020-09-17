function [identify, precision, recall,  f, cl_errors] = identify_and_compare(Ls, Lap, gamma_hats, gamma_cut, k)
    identify = zeros(k,1);
    cl_err = inf*ones(k,1);
    for i = 1:k
        W = diag(diag(Ls(:,:,i))) - Ls(:,:,i);
        W(W<0.001) = 0;
        Ls(:,:,i) = diag(sum(W)) - W;
        for j = 1:k
            er = norm(gamma_hats(:,i) - gamma_cut(:,j),'fro');
            %disp([string('Error for pairing i with j'),i, j, er])
            if (cl_err(i)>er)
                cl_err(i) = er;
                identify(i) = j;
            end
        end
    end

    for i = 1:k
        [precision(i,1), recall(i,1), f(i,1), NMI_score(i), num_of_edges(i)] = graph_learning_perf_eval(squeeze(Lap(:,:,identify(i))), squeeze(Ls(:,:,i)));
        %[ precision(i,2), recall(i,2), f(i,2), opt_Lapl] = precisionRecall_gmm(squeeze(Lap(:,:,identify(i))), squeeze(Ls(:,:,i)));
    end
    cl_errors = diag((gamma_hats - gamma_cut(:,identify))'*(gamma_hats - gamma_cut(:,identify)));
end