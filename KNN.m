classdef KNN < handle
    % This is a slightly simplified Matlab class that implements the kNN
    % algorithm. 
    
    properties 
        features = []; 
        labels = [];
    end
    
    methods 
        function knn = KNN(features, labels)
            % features: this should be a matrix where each COLUMN is a
            %   feature vector
            % labels: vector of labels for each feature
            knn.features = features;
            knn.labels = labels;
        end
        
        %takes row vectors
        function result = predict(knn, features, k)
            differences = knn.features - repmat(features, 1, size(knn.features, 2));
            distances = sum(differences.^2, 1).^.5;
            [~,ind] = sort(distances, 2, 'ascend');
            votes = knn.labels(ind(1:k));
            
            %-----Question 2(a)--------
            [M,F] = mode(votes)
            if k == 1 || range(F) == 0
                result = votes(1);
            else
                result = M;
            end
           %---------------------------------
        end
        
        function plotRegions(knn, k)
            % Plots the classifier's decision regions (only 2D feature 
            % vectors are supported here). 
            % 
            % k: number of neighbours
            
            if size(knn.features, 1) ~= 2
                error('this plot function only works with 2D feature spaces')
            end
            
            l = unique(knn.labels);
            c = {'r', 'b', 'k'};
            c = c(1:length(l));
            
            range1 = [min(knn.features(1,:)), max(knn.features(1,:))];
            steps1 = range1(1) + diff(range1)*(-.2:1/40:1.2);
            range2 = [min(knn.features(2,:)), max(knn.features(2,:))];
            steps2 = range2(1) + diff(range2)*(-.2:1/40:1.2);
            
            figure, hold on
            for i = 1:length(steps1)
                for j = 1:length(steps2)
                    p = predict(knn, [steps1(i); steps2(j)], k);
                    plot(steps1(i), steps2(j), sprintf('%s.', c{l == p}))
                end
            end
            
            f1 = knn.features(1,:);
            f2 = knn.features(2,:);
            for i = 1:length(l)
                label = l(i);
                ind = knn.labels == label;
                plot(f1(ind), f2(ind), sprintf('%sx', c{i}), 'MarkerSize', 10)
            end
            
            xlim([min(steps1) max(steps1)])
            ylim([min(steps2) max(steps2)])
        end        
    end    
end


