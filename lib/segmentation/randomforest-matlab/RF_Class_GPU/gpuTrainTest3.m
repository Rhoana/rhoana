%Test GPU bagging
%reset(gpuDevice(1));

%x = [0.01:0.01:1; (1:1:100) .*  rand(1,100); (201:1:300) .*  rand(1,100); (100:-1:1) .*  rand(1,100); rand(1,100); rand(1,100).*1000]';
x = [(0.01:0.01:1) ; (1:1:100) .*  rand(1,100); (201:1:300) .*  rand(1,100); (100:-1:1) .*  rand(1,100); rand(1,100); rand(1,100).*1000]';
classes = [ones(1,30),ones(1,70)*2]';

forest_gpu = gpuTrainRF2(x, classes, 300, 5);

%extra_options.sampsize = [maxNumberOfSamplesPerClass, maxNumberOfSamplesPerClass];
%extra_options.DEBUG_ON = 1;
%forest = classRF_train(x, y, 300,5,extra_options);
forest = classRF_train(x, classes, 300, 5);