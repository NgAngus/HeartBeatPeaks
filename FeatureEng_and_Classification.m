load("ecg-knn-data")
% Algorithm isolating by peak width
negc = -ecg

% Isolating peaks in ECG signal for feature creation
[dips, negIndices] = findpeaks(negc, 'MinPeakProminence', 0.002);
beforePeak = zeros(1,148);
afterPeak = zeros(1,148);
before = 0;
after = 0;

negIndices = transpose(negIndices);
for i = 1:148
    for negIndex = negIndices
        if negIndex < peakIndices(i)
            before = negIndex;
        else
            after = negIndex;
            break
        end
    end
    beforePeak(i) = before;
    afterPeak(i) = after;
end

% For width feature, seperation can be done
% by magnitude of width
widths = afterPeak - beforePeak;
Pwidths = (widths > 24) & (widths < 75);
QRSwidths = widths < 25;
Twidths = widths > 74;

xMax = 2.5 % For plot (max value of x)
% scatterplot features A and B together
figure, hold on;
title("Peak Width vs. Peak Amplitude")
% Define amplitude to be the difference between
% peak and median value.
amp = zeros(1,148)
xlim([0 xMax])
for i = 1:148
    amp(i) = (ecg(peakIndices(i)) - median(ecg));
end

% For each peak, plot amplitude and width
for i = 1:148
    if peakLabels(i) == 1
        plot(amp(i), widths(i), 'rx', 'MarkerSize', 12)
    elseif peakLabels(i) == 2
        plot(amp(i), widths(i), 'gx', 'MarkerSize', 12)
    else
        plot(amp(i), widths(i), 'bx', 'MarkerSize', 12)
    end
end


% Area Under Peak Feature Creation
area = zeros(1,148);
for i = 1:148
    slope = (ecg(beforePeak(i))-ecg(afterPeak(i)))/(beforePeak(i)-afterPeak(i));
    yint = ecg(beforePeak(i)) - slope*(beforePeak(i));
    baseline = transpose(slope*(1:16200) + yint);
    area(i) =  (abs(trapz(beforePeak(i):afterPeak(i), (ecg(beforePeak(i):afterPeak(i)) - baseline(beforePeak(i):afterPeak(i))))))
end

% Plot Area vs. Amplitude of Peak
figure, hold on;
title('Using area under peak')
title("Peak Area vs. Peak Amplitude")
xlim([0 xMax])
for i = 1:148
    if peakLabels(i) == 1
        plot(amp(i), area(i), 'rx', 'MarkerSize', 12)
    elseif peakLabels(i) == 2
        plot(amp(i), area(i), 'gx', 'MarkerSize', 12)
    else
        plot(amp(i), area(i), 'bx', 'MarkerSize', 12)
    end
end

% Using features in KNN to classify peaks
feature1 = ecg(peakIndices)';
feature2 = widths;
%  pass in normalized features
knn = KNN([feature1/std(feature1); feature2/std(feature2)], peakLabels);
plotRegions(knn, 3);
title('KNN on Amplitude vs. Peak Widths')

feature2 = area;
%  pass in normalized features
knn = KNN([feature1/std(feature1); feature2/std(feature2)], peakLabels);
plotRegions(knn, 3);
title('KNN on Amplitude vs. Peak Areas')


