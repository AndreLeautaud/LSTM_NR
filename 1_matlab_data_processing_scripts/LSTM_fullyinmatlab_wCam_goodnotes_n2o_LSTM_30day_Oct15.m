% upload csv data
% M=readtable('All_gap_sites_June2.csv'); 
% head(M)

% CD - need to deal with Site and Treatment columns as strings. Date as
% save to .mat 
% save('sites100plusdays_june.mat','M') 

% work timeline %
% 8/3/20 -> added De_Rosa. and made a holdout data set split.  %
% 8/3/20 -> added Ma+CONV to holdout. Fixed De_Rosa datasets (dropped trailing NA values).
    % added output RMSE results to csv. % 
    
% x/xx/20 -> did above and worked.
 
% x/xx/20 -> Added find real N2O value index to the pre NN creation process.

% x/xx/20 ? -> add hidden unit loop iteration 

% TBD -> Get permissions on data sets! 
% TBD -> check distributions of data sets and further site info. See about
    % changing what is calibration, validation, holdout
% TBD -> fix the 'real value' index resample. 
  

% TBD -> BIG TEST. Iterate through other possible covariate 


% rain and wfps - seems two variables overlap a lot. But maybe it tells us
% a lot about the soil as well...? 


close all
clear all

% set random number generator
s = rng(1);

pwd # your current working directory

% You will need to set below to your working environment! 
cd = 'N:\Research\Conant\chris\Dorich_n2o\Cam_AI';   

% make sure the csv is in your workspace and it will also output some
% figures and csvs, so just be aware ~30 files will appear in that folder
% as well 

%% import data (saved in .mat format for faster loading)
load('sites100plusdays_june.mat') % April - sites100plusdays.mat, load('raw_import_2.mat') __newwfps
rawtab = M;  % AllgapsitesDataTablen2osdTotal;
size(rawtab)

% class of site and treatment - need to be converted 

% subset data to just sites with wfps 
% rawtab = rawtab(strcmp(rawtab.Site, {'datalibrarian.158'}), :);
rawtab = [rawtab(strcmp(rawtab.Site, {'datalibrarian.158'}), :); rawtab(strcmp(rawtab.Site, {'deantoni'}), :);
    rawtab(strcmp(rawtab.Site, {'dougherty.8'}), :); %rawtab(strcmp(rawtab.Site, {'Garland_TTT_vineyard'}), :);
    rawtab(strcmp(rawtab.Site, {'vanDelden_2018'}), :);rawtab(strcmp(rawtab.Site, {'guangdi.76.auto'}), :);
    rawtab(strcmp(rawtab.Site, {'De_Rosa_2018'}), :)]; % no
    % rain/rainirrigation data for De_Rosa
   
    
% holdout datasets (these are randomly selected for now) 
holdoutdata = [rawtab(strcmp(rawtab.Treatment, {'Grass.0N'}), :); rawtab(strcmp(rawtab.Treatment, {'50N'}), :);
    rawtab(strcmp(rawtab.Treatment, {'Till_Pasture'}), :);rawtab(strcmp(rawtab.Treatment, {'Ma+CONV'}), :)];
    % ;rawtab(strcmp(rawtab.Treatment, {'Ma+CONV'}), :)];

    holdoutdata([1879:1889],:) = []; 
    
% exclude holdout data sets 
    % these need to be redone!
rawtab = rawtab(283:height(rawtab),1:38);  % Grass.0N
rawtab([3277:3626],:) = [];   % 50N   rawtab([3277:3626],:) = []; 
rawtab([6927:7291],:) = [];   % Till_Pasture  rawtab([7657:8021],:) = [];
rawtab([9968:10859],:) = [];   % Ma+CONV

% first 16 DeRosa days have no n2o data 
% last 11 DeRosa days have no climate data 
    % delete these days out to get gap-filling working % 
% Co+CONV
    % rawtab([7292:],:) % start actually fine for this one 
rawtab([10849:10859],:) = []; % Ma+Rd    
rawtab([9957:9967],:) = []; % CONV
rawtab([9065:9075],:) = []; % Co+Rd
rawtab([8173:8183],:) = []; % end Co+CONV

 % csvwrite('lstm_datasets.csv',rawtab)   
  
  % csvwrite('lstm_datasets_holdout.csv',holdoutdata)   
  
% rawtab = rawtab(strcmp(rawtab.Treatment{:,:},'Grass.0N'),2,:)
% rawtab = rawtab(strcmp(rawtab.Treatment,'Grass.0N') )      % ,2),:)
    % strcmp(rawtab.Treatment,'Grass.0N')
    % rawtab(ismember(rawtab,'Grass.0N')) = [];
    % idx = strcmp(rawtab.Treatment,'Grass.0N');
    % rawtab = rawtab(idx)
    % rawtab(:,idx) = [];
% rawtab = rawtab(~all(strcmp(rawtab{:,:},'Grass.0N'),2),:)

% [rawtab(strcmp(rawtab.Treatment, {'Grass.0N'}), :); rawtab(strcmp(rawtab.Treatment, {'50N'}), :);
%    rawtab(strcmp(rawtab.Treatment, {'Till_Pasture'}), :)];

% rawtab(strcmp(rawtab.Treatment{'Grass.0N'},1),:) = []; 

size(rawtab) 

site = string(rawtab.Site(2:end));
treatment = string(rawtab.Treatment(2:end));

%% Working on variable importance 
    % https://www.mathworks.com/matlabcentral/answers/181459-parameter-importance-for-neural-network
    % https://www.mathworks.com/matlabcentral/answers/44205-how-to-choose-the-most-significant-variables-from-possible-57-variables-for-neural-network-input
    % https://www.researchgate.net/post/How_to_determine_the_Importance_of_variables_in_neural_network_by_using_nntool_in_MATLAB
    % 
    
    % response surface plots 
    
%% get a list of unique site+treatment combinations
sites = site;
for i = 1:length(site)
    sites(i) = categorical([string(site(i))+'_~_'+string(treatment(i))]);
end

sites = unique(sites)
% subset to just sites with wfps 

 

%% parse out rain, n2o, and co2 time series data from each treatment
raincell = cell(length(sites),1);
n2ocell = cell(length(sites),1);
%co2cell = cell(length(sites),1);
maxlength = -inf;

%soilMcell = cell(length(sites),1);
wfpscell = cell(length(sites),1);

% CD - want to iterate through this next time around. 
    % 1 - remove CO2 (will want to try this both ways, not guaranteed that
    % we will have CO2 available for use, so want a NN without CO2. CO2 is
    % a valuable variable though, so want a scenario if CO2 was measured)
    % 2 - go through scenarios looking at other covariates (NH4, NO3, WFPS,
    % soilM, soiltemp5, tavg) - these are fairly common variables
    % (especially soilM/WFPS and temperatures) - what is the best NN that
    % can be developed using rainirrigation,soil moisture/temperature
    
for i = 1:length(sites)
    sitename = sites(i);
    sitename = strsplit(sitename,'_~_');
    treatment = sitename(2);
    sitename = sitename(1);
    
    data = rawtab((rawtab.Site == sitename) & (rawtab.Treatment == treatment),:);
    
    % we have NaN values for all covariates in a lot of these datasets
    % after a certain point. Were drawn out to 365 days...? 
    % if NaN across covariates, drop it 
        % data(all(~isnan(data),2),:); % this doesnt work, also looks
        % across all columns, only want to look at covariates being used in NN columns 
    
    if size(data,1) > maxlength
        maxlength = size(data,1);
    end
    
    %n2o
    n2ocell{i} = data.n2o;
    
    % use rainirrigation where rain is not available (and vice versa)
    rain = data.rain;
    irrigation = data.rainirrigation;
    k = rain;   % nan*ones(size(rain)); - what was this doing, just creating a NaN array? seemed to introduce NaN values... 
    k(isnan(irrigation)) = rain(isnan(irrigation));
    k(isnan(rain)) = irrigation(isnan(rain));
    raincell{i} = k;
    
    %co2
    %co2cell{i} = data.CO2;
    
    % wfps/soilM 
    wfpscell{i} = data.WFPS;
    
end


% access via n2ocell{2}(1:10)  - second site-treatment, first 10 n2o values

%% zero padding and matrix conversion (easier format to handle)
n2o = [];
rain = [];
%co2 = [];
wfps = []; 

for i = 1:length(n2ocell)
    
    n2odat = n2ocell{i};
    raindat = raincell{i};
    %co2dat = co2cell{i};
    wfpsdat = wfpscell{i}; 
    
    if length(n2odat) <  maxlength
        n2odat = [n2odat;nan*ones(maxlength-length(n2odat),1)];
        raindat = [raindat;nan*ones(maxlength-length(raindat),1)];
        %co2dat = [co2dat;nan*ones(maxlength-length(co2dat),1)];
        wfpsdat = [wfpsdat;nan*ones(maxlength-length(wfpsdat),1)];
    end
    n2o = [n2o,n2odat];
    rain = [rain,raindat];
    %co2 = [co2,co2dat];
    wfps = [wfps,wfpsdat];
    
end

%% remove barton.79 because it is all negative and doesn't make sense

% idx = 7:10; % BARTON INDICES HARDCODED RIGHT NOW
% n2o(:,idx) = [];
% rain(:,idx) = [];
% co2(:,idx) = [];
% wfps(:,idx) = [];

% only use datalibrarian.158, deantoni, dougherty.8, Garland_TTT_vineyard, guangdi.76.auto, vanDelden_2018 sites (only ones with wfps) 
    % Daqi working on unit conversion for soilM (bel.18, datalibrarian.137, De_Rosa_2018, dougherty.8, kelly.24.27, kelly.30.33, 
    % kelly.38, kelly.52, Mumford_2019, quayle.10, quayle.39, Rowlings.34)

  % idx = ; 

%n2o(:,idx) = [];
%rain(:,idx) = [];
%co2(:,idx) = [];
%wfps(:,idx) = [];

%% format
n2o = n2o';     % ' means transpose
rain = rain';
%co2 = co2';
wfps = wfps';

% fill internal gaps in n2o (I do this right now because otherwise there is
% insufficient continuous data to train with using both rain and co2)

% CD - need to think on this more, this may be problematic within our
% Gapfilling idea... 
for i = 1:size(n2o,1)
    a = n2o(i,:);
    b = 0.5 * (fillmissing(a, 'previous') + fillmissing(a, 'next'));
    n2o(i,:) = b;
end

for i = 1:size(rain,1)
    a = rain(i,:);
    b = 0.5 * (fillmissing(a, 'previous') + fillmissing(a, 'next'));
    rain(i,:) = b;
end

%for i = 1:size(co2,1)
%    a = co2(i,:);
%    b = 0.5 * (fillmissing(a, 'previous') + fillmissing(a, 'next'));
%    co2(i,:) = b;
%end

for i = 1:size(wfps,1)
    a = wfps(i,:);
    b = 0.5 * (fillmissing(a, 'previous') + fillmissing(a, 'next'));
    wfps(i,:) = b;
end

%% split on nan to form continuous segments for training set
c = cell(0);

for i = 1:size(n2o,1)
    seq = [];
    for j = 1:size(n2o,2)
        n2oval = n2o(i,j);
        rainval = rain(i,j); %rain or rainirrigation 
        %co2val = co2(i,j);
        wfpsval = wfps(i,j);
        
        split = isnan(n2oval)||isnan(rainval)||isnan(wfpsval); % split anywhere we have nan n2o, nan co2, or nan rain
        if ~split
            seq = [seq,[n2oval;rainval;wfpsval]];
        end
        if split&&(length(seq)>0)
            c = [c;seq];
            seq = [];
        end            
    end
end
    
seqlengths = zeros(length(c),1);
hasnan = zeros(length(c),1);;

for i = 1:length(c)
    seqlengths(i) = length(c{i});
    hasnan(i) = sum(isnan(c{i}(:)));
end

datacount = zeros(max(seqlengths),1);

for i = 1:max(seqlengths)
    datacount(i) = sum(seqlengths(seqlengths>=i));
end

% c and seqlenghts use 13 (instead of 17 sites?). Is Guangdi getting dropped out due to NAs? 
    % but there are only 3 guangdi in rawtab... 

 %% Notes on 3/2/21 
    % unsure about lines 259-297. Splitting a site-treatment based on NA
    % values..? 
    
    % De_Rosa datasets are the longest (881 days) 
    
    % n2odat =  881x1 frame 
        % the last site-treatment in the loop, expanded (NAs) to 881 length
    % n2o = 21x881 frame
        % frame containing all the site-treatments, with the expanded
        % length 
        % may have gap-filling in small gaps. need to check. (where first/last n2o values
        % were available, ie, if its at start/end don't think it would
        % gap-fill) 
            
    % c [18,1 cell]
        % c{1} gives you the first cell/array. We have it set up as a 3 channel cell/array/dataframe 
            % channels are: n2o, rain, WFPS
            
        % e.g., c{1} has 348 days of data. This is De_Rosa_2018_~_Co+CONV
        % e.g., c{18} is "vanDelden_2018_~_Pasture" where there are 730
        % days of measurements 

        % d = the training dataset (12 site-treatments)
            % then there are 1639 sequences in XTrain. More details on what
            % is happening here/how these were built 
        % dvalidation = the validation dataset (6 site-treatments)
            % their size is their respective number of days (and are
            % gap-filled?) 
        
        % e.g., d{1} is "vanDelden_2018_~_Forest" with 730 days of data  
        
        
        % XTrain - the training data, broken into sequence lengths 
            % 1639 sequences of 30 days 
          
        % XValidation - the validation data, broken into sequence lengths 
            % 763 sequences of 30 days 
        
        % question to Cam, all this correct? : 
            % there is no site-treatment identifier to xtrain.
            % and the model only sees the 30 day period at a time. 
            
            % given 30 continuous days of WFPS and rain data, with n2o
            % provided every 5th day -> the model predicts n2o for all 30
            % days 
            
%% how are continuous sequence lengths distributed?
figure
plot(datacount)
title('# of datapoints in sequences of length x or greater')
xlabel('x')

%% cull short sequences and get ready to split into short sequences of fixed length

% the LSTM looks at data within the sequence length. Does not look across
% sequences... So, want to make a sequence long enough to be relevant and
% provide enough info, but not too long to make it irrelevant... 
    % still working on this idea and thinking 
    
k = 30; % 30; %target sequence length
    % increasing k from 5 to 15, but will still provide n2o every 5 days.
    % So n2o on days 1,6,11 over the 15 days instead of looking at just day
    % 1 in a 5 day span 
    
% sampdays = 3; % number of days that actual n2o value is given to net as input (woops, not actually used in this version)
  
aug = 10; %augmentation factor (expand data by randomy window shifting by approximately this much)
    % still need to see what this is doing in practice more.. 
d = cell(0);

% CK - here I am only keeping sequences shorter than k
for i = 1:length(c)
    seq = c{i};
    if length(seq)>=k
        d = [d;seq];
    end
end

% CK - here I am randomly shuffling the sequences of length >= k
d = d(randperm(length(d),length(d)));

trainfrac = 0.7; %approximate fraction of data to be used for training
splitidx = floor(length(d)*trainfrac);

% CK - here I split off a separate array, dvalidation, for validation
% d will be used for training
dvalidation = d(splitidx+1:end);    % = 6 cell arrays, 30%
d(splitidx+1:end) = [];             % = 12 cell arrays, 70%

%% augment and make training set
XTrain = cell(0);
YTrain = cell(0);

rng(s); % initialize rng for consistency 
for i = 1:length(d) % for every entry of d
    count = 0; % start counter at zero
    seq = d{i}; % use the ith entry of d to draw a subsequence from
    
    % k is sequence length 
    while count < (length(seq)/k*aug) % this condition is met when we have enough subsequences to satisfy aug
        idx = randi([1,length(seq)-k+1],1); % pick a random starting index within seq
        idx = idx:(idx+k-1); % augment this idx to an interval of indicies of length k
        
        splitseq = seq(:,idx); % slice out the portion of seq corresponding to the indices in idx
        
        % sparsify n2o data % 
        sampidxn2o = zeros(1,k);
        % sampidxn2o = zeros(1,k); % creates vector - 1 * k vector of zeros. this artificially sparsifies our continuous n2o input
        
        % 1 - n2o data is used. 0 - n2o data not used
            % sampidxn2o(1:sampdays) = 1; % lets first sampdays to be used 
            % sampidxn2o(1:3:k) = 1; % would use 1,4,7 up till k
        sampidxn2o(1:5:k) = 1; 
        
        % only want to use real/measured n2o values 
        % (not simple gap-filled values)
            % so, if the n2o value is NaN, need to choose another day for
            % n2o in that sampidxn2o...
        
        
        %co2seq = seq(2,idx); % grab the co2 for this subsequence
        rainseq = seq(2,idx); % grab the rain for this subsequence
        wfpsseq = seq(3,idx); % grab the wfps for this subsequence
        
        %sampidxrain = ~isnan(co2seq); % compute when (for this subsequence) measurements were present -- unused here
        
        %co2seq(isnan(co2seq)) = 0; % set nan values in c02 to zero (could also do a sampidx type thing for this, but don't right now)
        % wfpsseq(isnan(wfpsseq)) = 0;
        
        % turned off before 6/19 run. Should be handled above
            % wfpsseq = fillmissing(wfpsseq,'linear'); 
        
        XTrain = [XTrain;[splitseq(1,:).*sampidxn2o;rainseq;wfpsseq;~sampidxn2o]]; % given first n2o point, co2, rain, and a list of where n2o is not sampled
        YTrain = [YTrain;splitseq(1,:)]; % we want the network to predict continuous n2o
        count = count + 1;
    end
end


%% make validation set
XValidation = cell(0);
YValidation = cell(0);

rng(s);
for i = 1:length(dvalidation)
    count = 0;
    seq = dvalidation{i};
    while count < (length(seq)/k*aug)
        idx = randi([1,length(seq)-k+1],1);
        idx = idx:(idx+k-1);
        
        % note in dataset which n2o values to use
        splitseq = seq(:,idx);
        sampidxn2o = zeros(1,k);
        sampidxn2o(1:5:k) = 1;
        
        %co2seq = seq(2,idx);
        rainseq = seq(2,idx);
        wfpsseq = seq(3,idx);
        
        %sampidxrain = ~isnan(co2seq);
        %co2seq(isnan(co2seq)) = 0;
        rainseq(isnan(rainseq)) = 0;
        % wfpsseq(isnan(wfpsseq)) = 0;
        wfpsseq = fillmissing(wfpsseq,'linear');
        
        XValidation = [XValidation;[splitseq(1,:).*sampidxn2o;rainseq;wfpsseq;~sampidxn2o]];
        YValidation = [YValidation;splitseq(1,:)];
        count = count + 1;
    end
end

data = XValidation;


%% machine learning (LSTM) - this is where training happens, will take some time to finish (~20-30 minutes) 
%normalize input data (VERY IMPORTANT)
mu = mean([XTrain{:}],2);
sig = std([XTrain{:}],0,2);

for i = 1:numel(XTrain)
    XTrain{i} = (XTrain{i} - mu) ./ sig;
end

for i = 1:numel(XValidation)
    XValidation{i} = (XValidation{i} - mu) ./ sig;
end
    

numResponses = size(YTrain{1},1); % to keras or tensorflow
featureDimension = size(XTrain{1},1);

numHiddenUnits = 200; % need to see about testing variations here. 
    % how was this selected? Is this 200 hidden neurons? Seems high... 
    % I like Hobs' answer: https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    
% CD - will want to iterate through numHiddenUnits to get stats on what
% value is the best 

% define network architecture 
layers = [ ...
    sequenceInputLayer(featureDimension) % 4 channels (n2o, rain, WFPS, sequence/selection)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(50) % what is this ? 
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% define training options
    % try variations here... 
maxEpochs = 500;
miniBatchSize = 10; 

% LSTM in python - batchsize, timestep, featureDimensions (sequencelength)

% CD - remind me on what miniBatchSize is? 
% CK - miniBatchSize is the number of training examples used to compute a
% gradient descent iteration (we combine all miniBatchSize gradients into
% one). This way we essentially average over several examples, so the
% gradient we descend is less noisy.


options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.001, ...
    'GradientThreshold',1, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation},...
    'Plots','training-progress',...
    'Verbose',0);

% train network
HUallRMSE = cell(0);

%xx=1
% iterate through hidden layers: (by steps of 25 for now)
%for hu = 25:25:200 
 %   
 %   numHiddenUnits = hu; 
 %   
 %   layers = [ ...
 %   sequenceInputLayer(featureDimension)
 %   lstmLayer(numHiddenUnits,'OutputMode','sequence')
 %   fullyConnectedLayer(50) % what is this, should we iterate here too? 
 %   dropoutLayer(0.5)
 %   fullyConnectedLayer(numResponses)
 %   regressionLayer];
 % think that is the only part changed with hidden unit change. 
 
 % this is where the model actually runs 
net = trainNetwork(XTrain,YTrain,layers,options);
%xx=xx+1

  % RMSE 
% rel_error = abs(sum(n2oseqq)-sum(Yest{1}))/sum(n2oseqq);
% HUallRMSE{xx} = rel_error;

 %end 

% how do I best compare the hu iterations and choose which one is the best? 
% choose the best model 


% XXX was the best model, run again and continue: 
% numHiddenUnits = hu; 
% layers = [ ...
%   sequenceInputLayer(featureDimension)
%    lstmLayer(numHiddenUnits,'OutputMode','sequence')
%    fullyConnectedLayer(50) % what is this, should we iterate here too? 
%    dropoutLayer(0.5)
%    fullyConnectedLayer(numResponses)
%    regressionLayer];

% [net,info] = trainNetwork(XTrain,YTrain,layers,options);
    % using [net,info] instead of net will allow for access to NN
    % structural info (results) 
    % https://www.mathworks.com/help/deeplearning/ref/trainnetwork.html#d117e55048
    % will likely slow it down. Will it still produce model to object net
    % and otherwise work fine..? 
    
    
%% save
    
modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
    save(['LSTM_hu200_SL30_3n2o' modelDateTime '.mat'],'net');


 
%% test network (LSTM) on random subset of validation set
YPred = predict(net,XValidation,'MiniBatchSize',1);
rng(s);
idx = randperm(numel(YPred),49);
rng(s);
figure

for i = 1:numel(idx)
    subplot(7,7,i)
    
    plot(YValidation{idx(i)},'--')
    hold on
    plot(YPred{idx(i)},'.-')
    samps = data{idx(i)}(1,:);
    samps(samps==0) = nan;
    plot(samps,'*')
    hold off
    
    rel_error = abs(sum(YValidation{idx(i)})-sum(YPred{idx(i)}))/sum(YValidation{idx(i)});
    
    title("Test Observation " + idx(i))
    xlabel("Time Step")
    ylabel("Value")
end
legend(["Test Data" "Predicted"],'Location','northeast')     

size(XValidation)

%% Chris added -> see stats on validation 

    % how would we use this NN that utilizes one day for every 30 days 

%% plotting just one site 
    % looking to have a NN model developed above that we can then use to
    % run on a site-treatment, and provide us a prediction for the complete time
    % series. 
    
ind1 = M(strcmp(M.Site, 'deantoni'),:); 
ind1 = ind1(strcmp(ind1.Treatment, 'CNT'),:);
ind1 = ind1(1:331,1:37); 

% ind1 = ind1(~any(ismissing(ind1),2),:);

%co2seq = ind1.CO2; % seq(2,idx);
rainseqq = ind1.rainirrigation(1:331); % seq(3,idx);
    
    % practice: wfpsseq = ind1.WFPS(208:220);
wfpsseqq = ind1.WFPS(1:331);    
   wfpsseqq = fillmissing(wfpsseqq, 'linear');
  
n2oseqq = ind1.n2o(1:331); 
    % n2oseqq = fillmissing(n2oseqq, 'linear'); % with this off it will
    % break, for now at least 

size(ind1) 

splitseqq =  transpose([n2oseqq,rainseqq,wfpsseqq]);  % seq(:,idx);

sampidxn2oq = zeros(1,height(ind1));
sampidxn2oq(1:5:height(ind1)) = 1; % use every 5th sample. This should still work

% find sampidxn2oq where index is met and n2o is NaN
    nalocs = n2oseqq(find(sampidxn2oq<1.1 & sampidxn2oq>0.9));    
    nalocs = find(isnan(nalocs))
    
% where n2o value is NaN and its been sampled, replace that point. 
for i = (1:15:height(ind1)-15 )   %1:numel(XDataset)
   % sampidxn2oq(i:i+14) 
   % n2oseqq(i:i+14)
    
   % first sample  
   if(isnan(n2oseqq(i)) == 1)
       first_non_NaN_index_of_X = find(~isnan(n2oseqq(i:i+14)), 1);
       % change indexes 
       sampidxn2oq(i) = 0;
       sampidxn2oq(i+first_non_NaN_index_of_X-1) = 1;
   end
   
   % 2nd     
   if(isnan(n2oseqq(i+5))==1)
       first_non_NaN_index_of_X = find(~isnan(n2oseqq(i+5:i+14)), 1);
       
   % change indexes 
   sampidxn2oq(i+5) = 0;
   sampidxn2oq(i+first_non_NaN_index_of_X-1) = 1;
   end
   
   % 3rd 
   if(isnan(n2oseqq(i+10))==1)
       sampidxn2oq(i+10) = 0;
       first_non_NaN_index_of_X = find(~isnan(n2oseqq(i+10:i+14)), 1,'first'); % go forward and try to find a sample
   
       % if still NaN
       if(isempty(first_non_NaN_index_of_X)==1)
        first_non_NaN_index_of_X = find(~isnan(n2oseqq(i+5:i+14)), 1,'last'); % go backward and try to find a sample
        sampidxn2oq(i+4+first_non_NaN_index_of_X) = 1; % real value in 6-10 range 
       else 
        sampidxn2oq(i+9+first_non_NaN_index_of_X) = 1; % real value in 11-15 range
       end         
       
   end

end 

% find sampidxn2oq where index is met and n2o is NaN
    nalocs = n2oseqq(find(sampidxn2oq<1.1 & sampidxn2oq>0.9));    
    nalocs = find(isnan(nalocs));
    nalocs

    sampidxn2oq(nalocs) = 0;
    
 % if there are NaN in data set it won't work 
    % the NaN values shouldn't be at sample points though...
     % are the NaN being used...? I don't want gap-filled values being used
     % in the net....
 n2oseqq = fillmissing(n2oseqq, 'linear');
 splitseqq =  transpose([n2oseqq,rainseqq,wfpsseqq]);  % seq(:,idx);

 rainseqq = transpose(rainseqq); 
 wfpsseqq = transpose(wfpsseqq);
  
% not validation, but the whole data set. 
XDataset = cell(0);
XDataset = [XDataset;[splitseqq(1,:).*sampidxn2oq;rainseqq;wfpsseqq;~sampidxn2oq]];
    % size(splitseqq),size(sampidxn2oq), size(rainseqq),size(wfpsseqq)
            % size(n2oseqq),
%XValidation = [XValidation;[splitseq(1,:).*sampidxn2o;co2seq;rainseq;wfpsseq;~sampidxn2o]];
% matrices aren't matching. Is this an NaN value issue? or due to
% sampidxn2o? 
    % size(ind1.n2o), 
    
 % scale it 
for i = 1:numel(XDataset)
    XDataset{i} = (XDataset{i} - mu) ./ sig;
end
% Given the whole dataset (n2o provided points set above), predict n2o 
    Yest = predict(net,XDataset,'MiniBatchSize',1);

figure
    plot(ind1.Date,ind1.n2o)
    
    hold on
    plot(ind1.Date,Yest{1})
    
    % put on the n2o training(?) points that were provided 
    samps = n2oseqq .* transpose(sampidxn2oq);
    %samps = data{idx(i)}(1,:);
    samps(samps==0) = nan;
    plot(samps,'*')
    
    xlabel("Date")
    ylabel("N2O emissions gN2O-N/ha-day")

    legend(["Observed" "Predicted"],'Location','northeast')   

    hold off % close figure  
    
    size(XDataset)
    
%% run NN for all site-treatments 
    % plot (time series with actual/estimated) results for each
    % calculate RMSE (and put to table) for model results 

    allRMSE = cell(0);
    
for i = 1:length(sites)
    sitename = sites(i);
    sitename = strsplit(sitename,'_~_');
    treatment = sitename(2);
    sitename = sitename(1);
    sites{i}
    
    ind1 = rawtab((rawtab.Site == sitename) & (rawtab.Treatment == treatment),:);
    
   % adjustments for issue cells  
    if sitename == "guangdi.76.auto"
        ind1 = ind1(110:height(ind1),1:width(ind1)); 
    end 
    if sites{i} == "dougherty.8_~_100N"
        ind1 = ind1(17:height(ind1),1:width(ind1));
    end 
    if sites{i} == "dougherty.8_~_25N"
        ind1 = ind1(17:height(ind1),1:width(ind1));
    end
    if sites{i} == "dougherty.8_~_50N"
        ind1 = ind1(17:height(ind1),1:width(ind1));
    end 
    
    %if sites{i} == "De_Rosa_2018_~_CONV"
    %    ind1 = ind1(1:364,1:width(ind1));
    %end
    
% ind1 = M(strcmp(M.Site, 'deantoni'),:); 
%ind1 = ind1(strcmp(ind1.Treatment, 'CNT'),:);
%ind1 = ind1(1:331,1:37); 
irrigation=ind1.rainirrigation; 
rain=ind1.rain;
%this isnt working, leaving NaN in places.. 
    k = rain;   
    % k = nan*ones(size(rain)); % creates cell of all NaN
    k(isnan(irrigation)) = rain(isnan(irrigation)); % in locations where rainirrigation is NaN, use rain value
    k(isnan(rain)) = irrigation(isnan(rain)); % in locations where rain is NaN, use rainirrigation value
rainseqq = k; 
    
wfpsseqq = ind1.WFPS;    
   wfpsseqq = fillmissing(wfpsseqq, 'linear');
  
n2oseqq = ind1.n2o; 
    n2oseqq = fillmissing(n2oseqq, 'linear');
    
splitseqq =  transpose([n2oseqq,rainseqq,wfpsseqq]);  % seq(:,idx);

sampidxn2oq = zeros(1,height(ind1));
sampidxn2oq(1:5:height(ind1)) = 1;

% find sampidxn2oq where index is met and n2o is NaN
    nalocs = n2oseqq(find(sampidxn2oq<1.1 & sampidxn2oq>0.9));    
    nalocs = find(isnan(nalocs))
    
% where n2o value is NaN and its been sampled, replace that point. 
for ii = (1:15:height(ind1)-15 )   %1:numel(XDataset)
   % sampidxn2oq(i:i+14) 
   % n2oseqq(i:i+14)
    
   % first sample  
   if(isnan(n2oseqq(ii)) == 1)
       first_non_NaN_index_of_X = find(~isnan(n2oseqq(ii:ii+14)), 1);
       % change indexes 
       sampidxn2oq(ii) = 0;
       sampidxn2oq(ii+first_non_NaN_index_of_X-1) = 1;
   end
   
   % 2nd     
   if(isnan(n2oseqq(ii+5))==1)
       first_non_NaN_index_of_X = find(~isnan(n2oseqq(ii+5:ii+14)), 1);
       
   % change indexes 
   sampidxn2oq(ii+5) = 0;
   sampidxn2oq(ii+first_non_NaN_index_of_X-1) = 1;
   end
   
   % 3rd 
   if(isnan(n2oseqq(ii+10))==1)
       sampidxn2oq(ii+10) = 0;
       first_non_NaN_index_of_X = find(~isnan(n2oseqq(ii+10:ii+14)), 1,'first'); % go forward and try to find a sample
   
       % if still NaN
       if(isempty(first_non_NaN_index_of_X)==1)
        first_non_NaN_index_of_X = find(~isnan(n2oseqq(ii+5:ii+14)), 1,'last'); % go backward and try to find a sample
        sampidxn2oq(ii+4+first_non_NaN_index_of_X) = 1; % real value in 6-10 range 
       else 
        sampidxn2oq(ii+9+first_non_NaN_index_of_X) = 1; % real value in 11-15 range
       end         
       
   end

end % end of real value resample 

% find sampidxn2oq where index is met and n2o is NaN
    nalocs = n2oseqq(find(sampidxn2oq<1.1 & sampidxn2oq>0.9));    
    nalocs = find(isnan(nalocs));
    nalocs

    sampidxn2oq(nalocs) = 0;
    
 % if there are NaN in data set it won't work 
    % the NaN values shouldn't be at sample points though...
     % are the NaN being used...? I don't want gap-filled values being used
     % in the net....
 n2oseqq = fillmissing(n2oseqq, 'linear');
 splitseqq =  transpose([n2oseqq,rainseqq,wfpsseqq]);  % seq(:,idx);

 rainseqq = transpose(rainseqq); 
 wfpsseqq = transpose(wfpsseqq);
  
% not validation, but the whole data set. 
XDataset = cell(0);
XDataset = [XDataset;[splitseqq(1,:).*sampidxn2oq;rainseqq;wfpsseqq;~sampidxn2oq]];
    % size(splitseqq),size(sampidxn2oq), size(rainseqq),size(wfpsseqq)
            % size(n2oseqq),  % size(XDataset{1})
    
 % scale it 
for t = 1:numel(XDataset)
    XDataset{t} = (XDataset{t} - mu) ./ sig;
end

% Given the whole dataset (n2o provided points set above), predict n2o 
    Yest = predict(net,XDataset,'MiniBatchSize',1);
    ind1.Yest = transpose(Yest{1});  % predict(net,XDataset,'MiniBatchSize',1);
    
 h=figure
 hold on
    plot(ind1.Date,ind1.n2o)
    
    plot(ind1.Date,Yest{1})
    
    % put on the n2o training(?) points that were provided 
    samps = n2oseqq .* transpose(sampidxn2oq);
    %samps = data{idx(i)}(1,:);
    samps(samps==0) = nan;
    plot(samps,'*')
    
    xlabel("Date")
    ylabel("N2O emissions gN2O-N/ha-day")

    legend(["Observed" "Predicted"],'Location','northeast')   
    title(sites{i})
    
 saveas(h, [sites{i} '_30DAY.png'], 'png')
    hold off % close figure
 %saveas(h, sprintf('FIG%d.png',sites{i}) );
    close(h)
    
    % RMSE 
    rel_error = abs(sum(n2oseqq)-sum(Yest{1}))/sum(n2oseqq);
    allRMSE{i} = rel_error;
    
    if(i==1)
        indall = ind1;
    else
        indall = vertcat(indall,ind1);
    end 
    
end  % end of site level loop 
 
%allRMSE = [sites,sites];
T = cell2table(allRMSE(:,:)) % cell2table(allRMSE(2:end,:),'VariableNames',allRMSE(1,:))
 
% Write the table to a CSV file
writetable(T,'allRMSE_30day.csv')
%dlmwrite('allRMSE.csv',((allRMSE)),'delimiter','')

% all of the daily data with the NN estimated n2o as well 
writetable(indall,'Dailydata_30day.csv')


%% run NN for the holdout site-treatment data sets 
    % plot (time series with actual/estimated) results for each
    % calculate RMSE (and put to table) for model results 

    allRMSEHOLDOUT = cell(0);
    
sitehold = string(holdoutdata.Site(2:end));
treatmenthold = string(holdoutdata.Treatment(2:end));

% get a list of unique site+treatment combinations
siteshold = sitehold;
for i = 1:length(sitehold)
    siteshold(i) = categorical([string(sitehold(i))+'_~_'+string(treatmenthold(i))]);
end

siteshold = unique(siteshold)

for i = 1:length(siteshold)
    sitename = siteshold(i);
    sitename = strsplit(sitename,'_~_');
    treatment = sitename(2);
    sitename = sitename(1);
    siteshold{i}
    
    ind1 = holdoutdata((holdoutdata.Site == sitename) & (holdoutdata.Treatment == treatment),:);
    
   % adjustments for issue cells  
    if sitename == "guangdi.76.auto"
        ind1 = ind1(110:height(ind1),1:width(ind1)); 
    end 
    if siteshold{i} == "dougherty.8_~_50N"
        ind1 = ind1(17:height(ind1),1:width(ind1));
    end 
irrigation=ind1.rainirrigation; 
rain=ind1.rain;
%this isnt working, leaving NaN in places.. 
    k = rain;   
    % k = nan*ones(size(rain)); % creates cell of all NaN
    k(isnan(irrigation)) = rain(isnan(irrigation)); % in locations where rainirrigation is NaN, use rain value
    k(isnan(rain)) = irrigation(isnan(rain)); % in locations where rain is NaN, use rainirrigation value
rainseqq = k; 
    
wfpsseqq = ind1.WFPS;    
   wfpsseqq = fillmissing(wfpsseqq, 'linear');
  
n2oseqq = ind1.n2o; 
    n2oseqq = fillmissing(n2oseqq, 'linear');
    
splitseqq =  transpose([n2oseqq,rainseqq,wfpsseqq]);  % seq(:,idx);

sampidxn2oq = zeros(1,height(ind1));
sampidxn2oq(1:5:height(ind1)) = 1;

% find sampidxn2oq where index is met and n2o is NaN
    nalocs = n2oseqq(find(sampidxn2oq<1.1 & sampidxn2oq>0.9));    
    nalocs = find(isnan(nalocs))
    
% where n2o value is NaN and its been sampled, replace that point. 
for ii = (1:15:height(ind1)-15 )   %1:numel(XDataset)
   % sampidxn2oq(i:i+14) 
   % n2oseqq(i:i+14)
    
   % first sample  
   if(isnan(n2oseqq(ii)) == 1)
       first_non_NaN_index_of_X = find(~isnan(n2oseqq(ii:ii+14)), 1);
       % change indexes 
       sampidxn2oq(ii) = 0;
       sampidxn2oq(ii+first_non_NaN_index_of_X-1) = 1;
   end
   
   % 2nd     
   if(isnan(n2oseqq(ii+5))==1)
       first_non_NaN_index_of_X = find(~isnan(n2oseqq(ii+5:ii+14)), 1);
       
   % change indexes 
   sampidxn2oq(ii+5) = 0;
   sampidxn2oq(ii+first_non_NaN_index_of_X-1) = 1;
   end
   
   % 3rd 
   if(isnan(n2oseqq(ii+10))==1)
       sampidxn2oq(ii+10) = 0;
       first_non_NaN_index_of_X = find(~isnan(n2oseqq(ii+10:ii+14)), 1,'first'); % go forward and try to find a sample
   
       % if still NaN
       if(isempty(first_non_NaN_index_of_X)==1)
        first_non_NaN_index_of_X = find(~isnan(n2oseqq(ii+5:ii+14)), 1,'last'); % go backward and try to find a sample
        sampidxn2oq(ii+4+first_non_NaN_index_of_X) = 1; % real value in 6-10 range 
       else 
        sampidxn2oq(ii+9+first_non_NaN_index_of_X) = 1; % real value in 11-15 range
       end         
       
   end

end % for resampling of missing points 

% find sampidxn2oq where index is met and n2o is NaN
    nalocs = n2oseqq(find(sampidxn2oq<1.1 & sampidxn2oq>0.9));    
    nalocs = find(isnan(nalocs));
    nalocs

    sampidxn2oq(nalocs) = 0;
    
 % if there are NaN in data set it won't work 
    % the NaN values shouldn't be at sample points though...
     % are the NaN being used...? I don't want gap-filled values being used
     % in the net....
 n2oseqq = fillmissing(n2oseqq, 'linear');
 splitseqq =  transpose([n2oseqq,rainseqq,wfpsseqq]);  % seq(:,idx);

 rainseqq = transpose(rainseqq); 
 wfpsseqq = transpose(wfpsseqq);
  
% not validation, but the whole data set. 
XDataset = cell(0);
XDataset = [XDataset;[splitseqq(1,:).*sampidxn2oq;rainseqq;wfpsseqq;~sampidxn2oq]];
    % size(splitseqq),size(sampidxn2oq), size(rainseqq),size(wfpsseqq)
            % size(n2oseqq),  % size(XDataset{1})
    
 % scale it 
for t = 1:numel(XDataset)
    XDataset{t} = (XDataset{t} - mu) ./ sig;
end

% Given the whole dataset (n2o provided points set above), predict n2o 
    Yest = predict(net,XDataset,'MiniBatchSize',1);
    ind1.Yest = transpose(Yest{1});  % predict(net,XDataset,'MiniBatchSize',1);
    
 h=figure
 hold on
    plot(ind1.Date,ind1.n2o)
    
    plot(ind1.Date,Yest{1})
    
    % put on the n2o training(?) points that were provided 
    samps = n2oseqq .* transpose(sampidxn2oq);
    %samps = data{idx(i)}(1,:);
    samps(samps==0) = nan;
    plot(samps,'*')
    
    xlabel("Date")
    ylabel("N2O emissions gN2O-N/ha-day")

    legend(["Observed" "Predicted"],'Location','northeast')   
    title(siteshold{i})
    
 saveas(h, [siteshold{i} '_holdout_30day.png'], 'png')
    hold off % close figure
 %saveas(h, sprintf('FIG%d.png',sites{i}) );
    close(h)
    
    % RMSE 
    rel_error = abs(sum(n2oseqq)-sum(Yest{1}))/sum(n2oseqq);
    allRMSEHOLDOUT{i} = rel_error;
    
    if(i==1)
        indall = ind1;
    else
        indall = vertcat(indall,ind1);
    end 
    
 end % for site level loop 
  
 T = cell2table(allRMSEHOLDOUT(:,:)) % cell2table(allRMSE(2:end,:),'VariableNames',allRMSE(1,:))
 
% Write the table to a CSV file
writetable(T,'allRMSEHOLDOUT_30day.csv')

% all of the daily data with the NN estimated n2o as well 
writetable(indall,'Dailydata_HOLDOUT_30day.csv')


    %% plot info... 
%figure

 %plot(YValidation{idx(i)},'--')
    %hold on
    %plot(YPred{idx(i)},'.-')
    %samps = data{idx(i)}(1,:);
    %samps(samps==0) = nan;
    %plot(samps,'*')
    %hold off 


%% run LSTM on all sites-treatments: plot results, get rel_error per site 

% create table to save relative_error+ stats from sites-treatments to
%error_table = table(site,treatment,relative_error); % some other sums or stats likely in here  
%vnames = {'site','treatment','relativeerror'};
%errorTable = array2table(zeros(0,3), 'VariableNames',vnames);

%vnamesParams = {'site','treatment','n2o','precip'}; % include the chosen covariates here
%AllNN = array2table(zeros(0,4), 'VariableNames',vnamesParams);

% run through site-treatments, run NN, get stats, save prediction data 
%for i = 1:length(sites)
    
%    sitename = sites(i);
%    sitename = strsplit(sitename,'_~_');
%    treatment = sitename(2);
%    sitename = sitename(1);
    
%    data = rawtab((rawtab.Site == sitename) & (rawtab.Treatment == treatment),:);
     % limdata = % pull out the data that the NN uses 
     
    % run NN to estimate n2o 
%    Yest = predict(net,limdata,'MiniBatchSize',1); % really want to cbind this result onto the table 

    % get relative error and write to table
%    rel_error = abs(sum(YValidation{idx(i)})-sum(YPred{idx(i)}))/sum(YValidation{idx(i)});
    
%        newRow = {sitename,treatment,rel_error}; 
%        error_table = append(error_table,newRow); 
    
     % save all n2o (and chosen covariates) data along with n2o prediction 
%     AllNN = [AllNN; limdata];
%end

% output error_table or write to csv 

%% some summary level plots 
  % - examining summary level data across site-treatments. Where is it working well. Where is it working poorly. 
  
    % 1 - skew and kurtosis (range of n2o values/distribution across sites) 
    % 2 - relative_error to sum of n2o (relative error to n2o sum across site)
    % 3 - relative error to skew/kurtosis (relative error to distribution across site)
    % 4 - climate vs n2o vs error (not a wide distribution in climate due
    % to mostly Australia, but sets up for next sites). 
    % ....
  
 %% some 'close-up' plots 
    % look at some specific scenarios where NN does/doesn't work well. 
    % need to start digging in to see 'why' 
    
 
 %% lets do that above first. 
 % then - will have to deal with other more categorical (not right word..)
 % variables, e.g., FertilizerkgNha, Tillagecm, crop (maybe) which we know should
 % show spikes in n2o after these, and without this info, the NN is likely
 % going to really struggle to predict these. 
 
 

