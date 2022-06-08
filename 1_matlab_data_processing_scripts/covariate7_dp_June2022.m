% 7 covariate script 

% values to edit ##########################
% SET sequence length
    % the amount of time (days) the NN can see at once 
    % iteration values of: 15, 30, 60 
SL = 30;

% SET n2o sample frequency (n2osf)
% sampdays = 3; % number of days that actual n2o value is given to net as input (woops, not actually used in this version)
    % iteration values of: 1, 3, 5, 7, 15
sampdays = 5;

covariates = 'rain_wfps_sT_fert_dayssinceapp';
% this is just for filename purposes, doesnt actually edit the covariate
    % pulling part, new script per model type needed 

% Model overview ##########################
    % base model = rain, wfps, n2o    (soiltemp?)
    
    % add temperature 
    % base + sT
    
    % with measured N 
    % base + NO3
    % base + sT + NO3

    % with implied N 
    % base + fertN(amt)
    % base + fertN(amt) + DaysSinceApp
    % base + fertN(amt) + DaysSinceApp + sT 

% other setup ###################
    cd 'C:\n2o_nn\LSTM_June2022\1_matlab_data_processing_scripts'

% DO - dropout layer gets added in the python training script 

% M=readtable('C:\n2o_nn\LSTM_June2022\gap_filled_rawtrainset_6and7covfertappanddayssince.csv'); %load this csv file(around 6 mb) into latlab drive and import it from there into the variable M
M=readtable('C:\n2o_nn\LSTM_June2022\GlobalN2ODB_NN_cleaned_dataset.csv');

s = rng(1);
rawtab = M; 
size(rawtab)
site = string(rawtab.SiteID(2:end));
treatment = string(rawtab.Treatment(2:end));

sites = site;
for i = 1:length(site)
    sites(i) = categorical([string(site(i))+'_~_'+string(treatment(i))]);
end

sites = unique(sites)

%% parse out covariates by treatment
% rain, wfps, sT, fert (N amt), dayssinceapp, 
    %   and y=n2o 
raincell = cell(length(sites),1);
n2ocell = cell(length(sites),1);
maxlength = -inf;

soilMcell = cell(length(sites),1); %we have to include both soil temp and soil inorganic nitrogen
%NO3 = cell(length(sites),1);
fert = cell(length(sites),1);
dayssinceapp = cell(length(sites),1);
wfpscell = cell(length(sites),1);
%datecell = cell(length(sites),1);
%sitecell = cell(length(sites),1);
%treatmentcell = cell(length(sites),1);
    
for i = 1:length(sites)
    sitename = sites(i);
    sitename = strsplit(sitename,'_~_');
    treatment = sitename(2);
    sitename = sitename(1);
    
    data = rawtab((rawtab.SiteID == sitename) & (rawtab.Treatment == treatment),:);
    
    data.rainirrigation = str2double(data.rainirrigation);
    data.DaysSinceApp = str2double(data.DaysSinceApp); % not 100% sure on this one... 
    data.Fertilizerapplied = str2double(data.Fertilizerapplied); % not 100% sure on this one... 

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
    soilMcell{i} = data.soiltemp10;
    %NO3{i} = data.NO3;
    dayssinceapp{i} = data.DaysSinceApp
    fert{i} = data.Fertilizerapplied;
    
    % wfps/soilM 
    wfpscell{i} = data.WFPS;

    %datecell{i} = datestr(data.Date)
    %sitecell{i} = data.SiteID
    %treatmentcell{i} = data.Treatment

    
end

n2o = [];
rain = [];
wfps = []; 
soil10 = [];
%soilno3 = [];
fertdayssinceapp=[];
fertapp = [];
%site = []
%treatment = []
%date = []
for i = 1:length(n2ocell)
    
    n2odat = n2ocell{i};
    raindat = raincell{i};
    wfpsdat = wfpscell{i}; 
    soil10dat = soilMcell{i};
    %soilno3dat = NO3{i}
    dayssincedat = dayssinceapp{i}
    fertappdat = fert{i}
    %sitedat = string(sitecell{i})
    %treatmentdat = string(treatmentcell{i})
    %datedat = (datecell{i})
    
    if length(n2odat) <  maxlength
        n2odat = [n2odat;nan*ones(maxlength-length(n2odat),1)];
        raindat = [raindat;nan*ones(maxlength-length(raindat),1)];
        wfpsdat = [wfpsdat;nan*ones(maxlength-length(wfpsdat),1)];
        soil10dat =  [soil10dat;nan*ones(maxlength-length(soil10dat),1)];
        %soilno3dat =  [soilno3dat;nan*ones(maxlength-length(soilno3dat),1)];
        dayssincedat =  [dayssincedat;nan*ones(maxlength-length(dayssincedat),1)];
        fertappdat =  [fertappdat;nan*ones(maxlength-length(fertappdat),1)];
        %sitedat =  [sitedat;nan*ones(maxlength-length(sitedat),1)];
        %treatmentdat =  [treatmentdat;nan*ones(maxlength-length(treatmentdat),1)];
        
        %datedat =  [datedat;nan*ones(maxlength-length(datedat),1)];
    end
    n2o = [n2o,n2odat];
    rain = [rain,raindat];
    wfps = [wfps,wfpsdat];
    soil10 = [soil10,soil10dat];
    %soilno3 = [soilno3,soilno3dat];
    fertdayssinceapp = [fertdayssinceapp,dayssincedat];
    fertapp = [fertapp,fertappdat];
    %site = [site,sitedat];
    %treatment = [treatment,treatmentdat];
    %date = [date, datedat];
end

n2o = n2o';     % ' means transpose? 
rain = rain';
wfps = wfps';
soil10 = soil10';
%soilno3 = soilno3';
fertdayssinceapp = fertdayssinceapp';
fertapp = fertapp';
%site = site';
%treatment = treatment';
%date= date';

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

for i = 1:size(wfps,1)
    a = wfps(i,:);
    b = 0.5 * (fillmissing(a, 'previous') + fillmissing(a, 'next'));
    wfps(i,:) = b;
end

for i = 1:size(fertapp,1)
    a = fertapp(i,:);
    b = 0.5 * (fillmissing(a, 'previous') + fillmissing(a, 'next'));
    fertapp(i,:) = b;
end

for i = 1:size(fertdayssinceapp,1)
    a = fertdayssinceapp(i,:);
    b = 0.5 * (fillmissing(a, 'previous') + fillmissing(a, 'next'));
    fertdayssinceapp(i,:) = b;
end
for i = 1:size(soil10,1)
    a = soil10(i,:);
    b = 0.5 * (fillmissing(a, 'previous') + fillmissing(a, 'next'));
    soil10(i,:) = b;
end

% 41 site-treatments,
    % 41x2163 is 41 site-treatments by the maximum length from one of the
    % site-treatments (2163 days) 

%% form continuous segments for training set
    % split on nan to
    % also randomly split by segments... 
c = cell(0);

for i = 1:size(n2o,1)
    seq = [];
    for j = 1:size(n2o,2)
        n2oval = n2o(i,j);
        rainval = rain(i,j); %rain or rainirrigation 
        wfpsval = wfps(i,j);
        soilval = soil10(i,j);
        %soilno3val = soilno3(i,j);
        dayssinceval = fertdayssinceapp(i,j);
        fertval = fertapp(i,j);

        %siteval = site(i,j);
        %treatmentval = treatment(i,j);
        %dateval = date(i,j);

        split = isnan(n2oval)||isnan(rainval)||isnan(soilval)||isnan(dayssinceval)||isnan(fertval)||isnan(wfpsval); % split anywhere we have nan n2o, nan co2, or nan rain
        if ~split
            seq = [seq,[n2oval;rainval;fertval;wfpsval;dayssinceval;soilval]];
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

% SET sequence length
% k = 30;
k = SL;

%k=15;  
    % increasing k from 5 to 15, but will still provide n2o every 5 days.
    % So n2o on days 1,6,11 over the 15 days instead of looking at just day
    % 1 in a 5 day span 
    
% sampdays = 3; % number of days that actual n2o value is given to net as input (woops, not actually used in this version)
    % sampdays = 7;

%augmentation factor
aug = 5; %augmentation factor (expand data by randomy window shifting by approximately this much)

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

% training data fraction 
    % how much data to be used in training of the model 
trainfrac = 0.7; %approximate fraction of data to be used for training
splitidx = floor(length(d)*trainfrac);

% CK - here I split off a separate array, dvalidation, for validation
    % d will be used for training
dvalidation = d(splitidx+1:end);    % = 11 cell arrays, 30%
d(splitidx+1:end) = [];             % = 24 cell arrays, 70%
    % this may be where validation got dropped out...? 

    
%% augment and make training set
XTrain1= cell(0);
YTrain1 = cell(0);

% d are windows (taken from site-treatment) with continuous data
% idx - the days that will make up the sampled segment of SL 
% splitseq - the idx sampled values take from the 'window' 
% for the last i (at least in my test, it was guangdi.76.auto, NoTill_Pea,  8/6/14 - 9/4/14)
    % do to random sorting, hard to identify which site-treatments are
    % where.... 

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
            % not sure we want/need this .... n2osf (sampdays) is supposed to handle
            % this...? 
        sampidxn2o = zeros(1,k);

        % sampidxn2o = zeros(1,k); % creates vector - 1 * k vector of zeros. this artificially sparsifies our continuous n2o input
        
        % 1 - n2o data is used. 0 - n2o data not used
            % sampidxn2o(1:sampdays) = 1; % lets first sampdays to be used 
            % sampidxn2o(1:3:k) = 1; % would use 1,4,7 up till k
        % sampidxn2o(1:7:k) = 1; 

        % set the n2o sample day 
            % think this is reversed right now... 
        sampidxn2o(1:sampdays:k) = 1; 
        
        % only want to use real/measured n2o values 
        % (not simple gap-filled values)
            % so, if the n2o value is NaN, need to choose another day for
            % n2o in that sampidxn2o...
        
        rainseq = seq(2,idx); % grab the rain for this subsequence
        fertseq = seq(3,idx);
        wfpsseq = seq(4,idx); % grab the wfps for this subsequence
        daysinceappseq = seq(5,idx);
        soil10seq = seq(6,idx); % grab the soil10 for this subsequence
        %soilno3seq = seq(4,idx); % grab the soil10 for this subsequence      
        
        %sampidxrain = ~isnan(co2seq); % compute when (for this subsequence) measurements were present -- unused here
        
        % wfpsseq(isnan(wfpsseq)) = 0;
        
        % integrate only real n2o value use 
            % NA n2o values were gap-filled. 
            % the below code checks and resamples n2osf based on whether
            % n2o is real for that day 
                % likely improves accuracy given the real world measured
                % values only, adds in some randomness associated with
                % n2osf (again likely better represents real world sampling
                % by techs) 
                % unsure what potential negative impacts would be... 

        % turned off before 6/19 run. Should be handled above
            % wfpsseq = fillmissing(wfpsseq,'linear'); 
        
        % XTrain1 =
        % [XTrain1;[splitseq(1,:).*sampidxn2o;rainseq;fertseq;wfpsseq;daysinceappseq;soil10seq;~sampidxn2o]];
        % why was the '~sampidxn2o' in place? and not just 'sampidxn2o'  ?
        % 0,1 values seemed reversed because of this ... 
        XTrain1 = [XTrain1;[splitseq(1,:).*sampidxn2o;rainseq;fertseq;wfpsseq;daysinceappseq;soil10seq;sampidxn2o]]; 
        YTrain1 = [YTrain1;splitseq(1,:)]; % we want the network to predict continuous n2o
        count = count + 1;
    end
end

%% raw values check 
    % QC opportunity here - are things doing what we want/expect? 

n2ocell % 41(site-treatments)x1, each cell will be length(days)x1(n2o)
    % so n2ocell{1} will give the first site-treatment 

size(XTrain1) % the number of SL that were created

splitseq % the last SL created 

XTrain1{size(XTrain1,1)} % grab the last SL from the SL array (should match the splitseq) 

% splitseq(1,:).*sampidxn2o;rainseq;fertseq;wfpsseq;daysinceappseq;soil10seq;~sampidxn2o]]
    % covariate order: n2o*n2osf(1,0), rain, fert, wfps, dayssinceapp, soiltemp10, n2osf

%% make validation set
    % don't think this is being used at all yet.... 

XValidation1 = cell(0);
YValidation1 = cell(0);

rng(s);
for i = 1:length(dvalidation)
    count = 0;
    seq = dvalidation{i};
    while count < (length(seq)/k*aug)
        idx = randi([1,length(seq)-k+1],1);
        idx = idx:(idx+k-1);
        
        splitseq = seq(:,idx);
        sampidxn2o = zeros(1,k);

            % sampidxn2o(1:5:k) = 1;
        % should use this: sampidxn2o(1:sampdays:k) = 1; % ...? 

        rainseq = seq(2,idx);
        %soilno3seq = seq(3,idx); % grab the soil10 for this subsequence
        wfpsseq = seq(4,idx);
        soil10seq = seq(6,idx);
        fertseq = seq(3,idx);

        daysinceappseq = seq(5,idx);
        
        %sampidxrain = ~isnan(co2seq);
        %rainseq(isnan(rainseq)) = 0;
        % wfpsseq(isnan(wfpsseq)) = 0;
        %wfpsseq = fillmissing(wfpsseq,'linear');
        %soilno3seq = fillmissing(soilno3seq,'linear');
        
        % 
        XValidation1 = [XValidation1;[splitseq(1,:).*sampidxn2o;rainseq;fertseq;wfpsseq;daysinceappseq;soil10seq;sampidxn2o]];
        YValidation1 = [YValidation1;splitseq(1,:)];
        count = count + 1;
    end
end

data = XValidation1;

%% normalize input data (VERY IMPORTANT)
    % mu and sigma from XTrain so its generalizeable...? 

% the last row(?) in the df is n2osf (whether or not the model will see that n2o sample, 1=yes, 0=no)
    % gets normalized to -2.236 and 0.4472 (at least in my test example).
    % Is this an issue? does it matter? 
mu = mean([XTrain1{:}],2);
sig = std([XTrain1{:}],0,2);

for i = 1:numel(XTrain1)
    XTrain1{i} = (XTrain1{i} - mu) ./ sig;
end

% not actually doing any normalize on the xval in the model build/estimation...? 
for i = 1:numel(XValidation1)
    XValidation1{i} = (XValidation1{i} - mu) ./ sig;
end

%% save everything 
% XTrain, Xvalid, YTrain, Yvalid, Mean, Sigma 

% fmt = 'XTrain_SL%0.2f_%03d_n2osf%03d.mat'; 
XT_fmt = 'XTrain_%s_SL%0.0f_n2osf%02d.mat'; 
YT_fmt = 'YTrain_%s_SL%0.0f_n2osf%02d.mat';
XV_fmt = 'XValid_%s_SL%0.0f_n2osf%02d.mat';
YV_fmt = 'YValid_%s_SL%0.0f_n2osf%02d.mat';
M_fmt = 'Mean_%s_SL%0.0f_n2osf%02d.mat';
S_fmt = 'Sigma_%s_SL%0.0f_n2osf%02d.mat';

XT_fname = sprintf(XT_fmt,covariates,SL,sampdays);
YT_fname = sprintf(YT_fmt,covariates,SL,sampdays);
XV_fname = sprintf(XV_fmt,covariates,SL,sampdays);
YV_fname = sprintf(YV_fmt,covariates,SL,sampdays);
M_fname = sprintf(M_fmt,covariates,SL,sampdays);
S_fname = sprintf(S_fmt,covariates,SL,sampdays);

% save .mat files (model train/test data) here:
cd 'C:\n2o_nn\LSTM_June2022\2_matlab_traintest_matdata'

% train 
save(XT_fname,"XTrain1")
save(YT_fname,"YTrain1")

% mean and sigma 
save(M_fname,"mu")
save(S_fname,"sig")

% valid
% these aren't currently being used 
   % question of generalizeability of the model? 
    % reverse standardization being done on holdout sites ... 
save(XV_fname,"XValidation1")
save(YV_fname,"YValidation1")
