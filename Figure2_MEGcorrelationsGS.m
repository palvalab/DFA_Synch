% Brain criticality predicts individual synchronization levels in human brains
% Demo code for MEG subject correlations of Figure 2 (MEG part)
% included explicitly for panels C, D, E, F; directions for the others (and also for Figure 3)
% from preprocessed intermediate MinDataset (connectivity matrices and DFA)

% Copyright (C) 2022 Marco Fuscà 
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
% See the GNU General Public License for more details.
% <http://www.gnu.org/licenses/>.

%% Dataset location
paths.rsdata = 'path_to_dataset_folder'; %path_to_dataset_folder (downloaded from Dryad; e.g. pwd or 'C:\Users\user\Downloads' or '/home/user/Downloads/')
paths.datasname = 'MEG_data.mat'; 

%% output 
paths.out = pwd; % %path_for_output_figures_and_results (e.g. pwd or mfilename('fullpath'))
paths.outname = sprintf('Figure2_%s',datestr(now,'mmdd_HH'));
paths.output = fullfile(paths.out, paths.outname);

%% load dataset
dataset = load(fullfile(paths.rsdata,paths.datasname)); %around 3GB for MEG

%% parameters
% frequencies & nparcels & permutations
dfreqs = dataset.frequencies; % = [2, 3, 4, 5, 7, 9, 10, 11, 12, 15, 20, 26, 32, 38, 45, 55, 65, 77, 90, 115, 135, 165, 185, 225]; % in Hz
nfreqs = numel(dfreqs); % = 24;
nParc = numel(dataset.parcels); % = 400; % n of MEG parcels
nperm = 100; % case resampling surrogate permutations ( = 1000 in the study)

% see also freqreg variable in section below %% Figure 2 E F (Single regressions)
% a single frequency wPLI-DFA regression across subjects to plot (7 and 10 Hz in paper)

%% aestetic plot x axis
df0 = [dfreqs(1)*0.9 dfreqs(end)*1.15];
dfxticks = [2 3 5 10 20 30 50 100 150 200 250];
dfxticksl = string([2 3 5 10 20 30 50 100 nan 200 nan]);
dfxminticks = [4 6 7 8 9 40 60 70 80 90 110:10:190 210:10:250];
    
%% Window and figure screen size
wscrize = get(0,'Screensize');
fscrize(1:2) = floor(wscrize(1:2) + (wscrize(3:4)/6));fscrize(3:4) =floor( wscrize(3:4) - (wscrize(3:4)/6))-fscrize(1:2);
fscrize2([1,3]) = wscrize([1,3]) +[5,-5]; fscrize2(2) = floor(wscrize(2) + (wscrize(4)/10)); fscrize2(4) =floor( wscrize(4) - (wscrize(4)/10))-fscrize(2);

%% get DFA exponents
dfaex = dataset.DFA;
nsubjs = size(dfaex,1); % = 192; %actually n sets

%% get connectivity matrices
connmetric = 'wPLI';
con_mats = dataset.(connmetric);

%% edges masks based on edge fidelity & cpPLV
% 15% mask rejected edges (based on around 10% cross-parcel edge fidelity and around 5%  cpPLV, a nParc*nParc matrix)
% masks of single subjs/sets edges computed individually (a nSbSet*nParc*nParc matrix)
efidmask = dataset.efidmask;
efidmaskstr = 'IEMask';

%% parcels masks
% same as edge masks above but for parcels (rejctd based on num of rejctd edges per parcel) 
sfidmask = dataset.sfidmask;
sfidmaskstr = 'IPMask';

%% Shuffle Subj order nperm times
permordn = zeros([nperm, nsubjs]); %case resampling (shuffling subjs)
for irpo =1:nperm, permordn(irpo,:) = randperm(nsubjs); end %can be saved for replication 

%% apply mask and mean connectivity
syncns = nan([nsubjs,nfreqs,nParc]);  %local conn (NS, for topographies -Figure 3- and within-subjs corrs) 
syncgsm = nan([nsubjs,nfreqs]); %global conn (GS)
for csi = 1:nsubjs
        for fmi = 1:nfreqs 
            cmetric = squeeze(abs(con_mats(csi,fmi,:,:)));
            cmetric(logical(eye(size(cmetric)))) = NaN;
            cmetric(squeeze(efidmask(csi,:,:))) = NaN; 
            syncns(csi,fmi,:) = mean(cmetric,'omitnan'); % simple channel mean (for Figure 3) (nanmean before 2020b)   
            syncgsm(csi,fmi)= mean(squeeze(syncns(csi,fmi,sfidmask(csi,:))),'omitnan');  % simple subject mean 
        end
end 

%% apply mask and mean dfa
dfdatams = nan([nsubjs,nfreqs]); for isf = 1:nsubjs, dfdatams(isf,:) = mean(squeeze(dfaex(isf,:,sfidmask(isf,:))),2); end

%% Distributions (Figure 2 A B)
% not included, but you can plot it already here with your favourite visualization
% MEG_wPLI_Distrbution = plot(mean(syncgsm)); %use bootci with mean for intervals
% MEG_DFA_Distrbution =  plot(mean(dfdatams));
% (have a look at https://github.com/Markwelt/mutils/blob/main/raincloud_plot.m)
% MEG_wPLI_Distrbution_singlefreq = raincloud_plot(syncgsm(:,frequency));
% MEG_DFA_Distrbution_singlefreq = raincloud_plot(dfdatams(:,frequency));

%% subjs connect-DFA Pearson correlation
[nonoutNl, corr_rhol, corr_pvall, corr_rhoLowl, corr_rhoUpl] = deal(nan([nperm+1,nfreqs]));
    for ifc = 1:nfreqs
        % simple corr returns corr_pval = 0 for very high rhos (so regstats taken instead)
        dfam = dfdatams(:,ifc); syncm = syncgsm(:,ifc);
        for dfsi = 1:nperm+1
            sync_r = syncm;
            if dfsi == 1
                dfam_r = dfam;
            else
                dfam_r = dfdatams(permordn(dfsi-1,:),ifc);
            end
            [sync_r, sync_routi] = rmoutliers(sync_r); dfam_r(sync_routi) = []; %rmoutliers in Matlab2021b+  otherwise can copy rmMissingOutliers('rmoutliers'...
            nonoutNl(dfsi,ifc) = numel(sync_r); %numel after removed outliers
            cregsrts = regstats(dfam_r,sync_r,'linear');
            corr_pvall(dfsi,ifc)= cregsrts.tstat.pval(2); if cregsrts.tstat.pval(2) == 0, corr_pvall(dfsi,ifc)= cregsrts.fstat.pval; end
            
            [R,~,RL,RU] = corrcoef(sync_r,dfam_r); % lower and upper bounds of correlation coefficient
            corr_rhoLowl(dfsi,ifc)= RL(2); corr_rhoUpl(dfsi,ifc)= RU(2);  
            corr_rhol(dfsi,ifc)= R(2); %   == cregsrts.beta(2)
        end
    end

%% Get real and permuted shuffled mean 
corr_pval(1,:) = corr_pvall(1,:);
% corr_pval(2,:) = 10.^mean(log10(corr_pvall(2:nperm+1,:))); %averaging p-values like this would be for illustration purposes only and not valid, see mafdr below

corr_rho(1,:) = corr_rhol(1,:);
corr_rho(2,:) = mean(corr_rhol(2:nperm+1,:));

nonoutN(1,:) = nonoutNl(1,:); nonoutN(2,:) = mean(nonoutNl(2:nperm+1,:));
corr_rhoLow(1,:) = corr_rhoLowl(1,:); corr_rhoUp(1,:) = corr_rhoUpl(1,:);

for ifc = 1:nfreqs
    pCIs = prctile(corr_rhol(2:nperm+1,ifc),[2.5,97.5]);
    corr_rhoLow(2,ifc) = pCIs(1);
    corr_rhoUp(2,ifc) = pCIs(2);
end

%% Subjs Corrs FDR correction
[corr_fdr] = mafdr(corr_pval(:)); %BHFDR more conservative, especially with large amount of data
[corr_fdrb] = mafdr(corr_pval(:),'BHFDR',true);  % especially if you decide to add a mean surrogate pvalues measure too
% mafdr in bioinfo toolbox (can be copied)
% for permutation-derived pvalues, use invprctile by Durga Lal Shrestha or [muH,sigmaH] = normfit(distribution); pval = 1-abs(normcdf(value,muH,sigmaH));

%% save interim data
% optionally, since depending on nperms it may take a while and may need to redo the figures
% save([paths.output '_' num2str(nperm) 'nperms.mat'],'corr_*','nonoutNl','permordn')

%% Figure 2 C (subjs correlation plot)
plcolormain = [0 .5 .8; .7 .7 .7]; plcolorfaint = [.7 .8 1]; plcolorfainter = [.8 .8 .8]; % sync & dfa real
PerSpear= 'Pearson';

figfptitl = 'Fig2C MEG wPLI-DFA Correlations';
figfpcorr = figure ('Name',figfptitl,'PaperPositionMode','auto','Position', fscrize2, 'Color', [1 1 1]);

subplot(1,2,1)
    for dfsi = 2:-1:1  % dfa surrogate or real (real at the end to be on top)
    corr_rho_plt = squeeze(corr_rho(dfsi,:));
    lowbounds = squeeze(corr_rhoUp(dfsi,:));
    upbounds = squeeze(corr_rhoLow(dfsi,:));
    plot(dfreqs,lowbounds,'Color', plcolormain(dfsi,:), 'LineWidth',0.5); % plot bounds (consider also shadedErrorBar.m by Rob Campbell)
    hold on
    plot(dfreqs,upbounds,'Color', plcolormain(dfsi,:), 'LineWidth',0.5); % plot bounds
    plot(dfreqs,corr_rho_plt,'Color', plcolormain(dfsi,:), 'LineWidth',2); % plot correlation coefficients across freqs   
    end
    
    if ~any(get(gca,'YLim')==0), plot(df0,zeros([1 2]),':k'); end
    ylabel([PerSpear ' \rho'])
    title(figfptitl,'interpreter','none')
    hax = gca;
    set(hax,'XLim',df0,'XTick',dfxticks,'XTickLabel',dfxticksl,'XMinorTick', 'on', 'XScale', 'log','FontSize',20,'FontName','Myriad Pro')
    set(hax.XAxis,'MinorTickValues', dfxminticks)
    xlabel('Frequency [Hz]')
    box off, hold off 
    
subplot(1,2,2)   
    log10_corr_pval = -log10(corr_pval);
    plot(log10_corr_pval,'.-','color',plcolorfaint)  
    hold on
    log10_corr_fdr = -log10(corr_fdr);
    plot(log10_corr_fdr,'.-','color',plcolorfainter)
    
    log10_corr_fdrb = -log10(corr_fdrb);
    plot(log10_corr_fdrb,'.-','Color',plcolormain(1,:)), hold on 
    plot(0:nfreqs+1,ones([1 nfreqs+2])*-log10(0.05),':k'); plot(0:nfreqs+1,ones([1 nfreqs+2])*-log10(0.01),':k'); plot(0:nfreqs+1,ones([1 nfreqs+2])*-log10(0.001),':k');
    
    ylims = get(gca,'YLim'); set(gca,'YLim',[0,ylims(2)]);
    ylabel([PerSpear ' p-value'])
    set(gca,'XLim',[0 nfreqs+1],'XTick',1:nfreqs,'XTickLabel',dfreqs)
    yticklabls = get(gca,'YTickLabel');  set(gca,'YTickLabel',['1';cellfun(@(x) ['10^{-' x '}'], yticklabls(2:end),'unif',0)])
    xlabel('Frequency [Hz]')
    for iptxt = 1:nfreqs
        postxt = log10_corr_fdrb(iptxt)+ ylims(2)/40; if ylims(2)-postxt < ylims(2)/20, postxt = log10_corr_fdrb(iptxt)-ylims(2)/40; end
        %if postxt == Inf, postxt = ylims(2)-ylims(2)/40; end %in case of pvals = 0
        text(iptxt,postxt,num2str(-log10_corr_fdrb(iptxt),2),'HorizontalAlignment','center')
    end

    title(['pvals ' efidmaskstr ' ' sfidmaskstr])
    box off
 
print (figfpcorr,'-dpng', '-r400', [paths.output '_DFA_wPLI_' PerSpear 'rpval.png']) %png
%     print (figfpcorr,'-dmeta','-painters', [paths.output '_DFA_wPLI_' PerSpear 'rpval.emf']) %emf
%     close(figfpcorr)


%% Partial Quadratic correlations

%% subjs connect-DFA Partial Quadratic correlations
[corrq_rhol, corrq_pvall, corrq_arsql] = deal(nan([nperm+1,nfreqs]));
[corrq_rhoLowl, corrq_rhoUpl, corrq_r2Lowl, corrq_r2Upl] = deal(nan([1,nfreqs]));

    for ifc = 1:nfreqs
        % simple corr returns corrq_pval = 0 for very high rhos (so regstats taken instead)
        dfam = dfdatams(:,ifc); syncm = syncgsm(:,ifc);
        for dfsi = 1:nperm+1
            sync_r = syncm;
            if dfsi == 1
                dfam_r = dfam;
            else
                dfam_r = dfdatams(permordn(dfsi-1,:),ifc);
            end
            [sync_r, sync_routi] = rmoutliers(sync_r); dfam_r(sync_routi) = []; %rmoutliers in Matlab2021b+ otherwise can copy rmMissingOutliers('rmoutliers',A)
            lcregsrts = regstats(dfam_r,sync_r,'linear');
            cregsrts = regstats(lcregsrts.r,sync_r,'quadratic');
            corrq_pvall(dfsi,ifc)= cregsrts.tstat.pval(2); if cregsrts.tstat.pval(2) == 0, corrq_pvall(dfsi,ifc)= cregsrts.fstat.pval; end
            corrq_rhol(dfsi,ifc)= cregsrts.beta(3); %  p1*x^2
            corrq_arsql(dfsi,ifc)= cregsrts.rsquare; % .adjrsquare / .rsquare            
                           
            if dfsi == 1
                poly2CI = confint(fit(sync_r,lcregsrts.r,'poly2')); %Curve Fitting Toolbox (no way around it, in case comment here and below)
                corrq_rhoLowl(ifc)= poly2CI(1,1); corrq_rhoUpl(ifc)= poly2CI(2,1);
                r2 = abs(corrq_arsql(dfsi,ifc)); nr2 = numel(sync_r); %always positive anyway (except maybe when using .adjrsquare)
                SEr2 = sqrt(((4*r2)*((1-r2)^2)*((nr2-2-1)^2))/(((nr2^2)-1)*(nr2+3)));
                r2CI = tinv(1-0.05/2,nr2-2-1)*SEr2;%  Cohen et al. (2003) Applied Multi Regr/Corr Anls for the Behav Sci
                corrq_r2Lowl(ifc) = r2-r2CI; corrq_r2Upl(ifc) = r2+r2CI;
            end
        end
    end

%% Get real and permuted shuffled mean part-quadr
corrq_pval(1,:) = corrq_pvall(1,:);

corrq_rho(1,:) = corrq_rhol(1,:);
corrq_rho(2,:) = mean(corrq_rhol(2:nperm+1,:));

corrq_arsq(1,:,:) = corrq_arsql(1,:,:);
corrq_arsq(2,:,:) = mean(corrq_arsql(2:nperm+1,:,:));

corrq_rhoLow(1,:) = corrq_rhoLowl(1,:); corrq_rhoUp(1,:) = corrq_rhoUpl(1,:);

for ifc = 1:nfreqs
    pCIs = prctile(corrq_rhol(2:nperm+1,ifc),[2.5,97.5]);
    corrq_rhoLow(2,ifc) = pCIs(1);
    corrq_rhoUp(2,ifc) = pCIs(2);
end

corrq_r2Low(1,:,:) = corrq_r2Lowl(:,:); corrq_r2Up(1,:,:) = corrq_r2Upl(:,:);

for ifc = 1:nfreqs
    pCIs = prctile(-sign(corrq_rhol(2:nperm+1,ifc)).*corrq_arsql(2:nperm+1,ifc),[2.5,97.5]);
    corrq_r2Low(2,ifc) = pCIs(1);
    corrq_r2Up(2,ifc) = pCIs(2);
end

%% Subjs Corrs FDR correction part-quadr
[corrq_fdr] = mafdr(corrq_pval(:)); %BHFDR more conservative, especially with large amount of data
[corrq_fdrb] = mafdr(corrq_pval(:),'BHFDR',true); % especially if you decide to add mean pvalues (results do not change)

%% save interim data part-quadr
% optionally, since depending on nperms it may take a while and may need to redo the figures
% save([paths.output '_' num2str(nperm) 'nperms_pquadr.mat'],'corrq_*')

%% Figure 2 D (subjs partial quadratic correlation plot)
plcolormain = [0 .5 .8; .7 .7 .7]; plcolorfaint = [.7 .8 1]; plcolorfainter = [.8 .8 .8]; % sync & dfa real
PerSpear= 'Partial Quadratic';

figfptitl = 'Fig2D MEG wPLI-DFA Partial Quadraric Corrs';
figfpcorr = figure ('Name',figfptitl,'PaperPositionMode','auto','Position', fscrize2, 'Color', [1 1 1]);

subplot(1,2,1)
    for dfsi = 2:-1:1  % dfa surrogate or real (real at the end to be on top)
    corrq_sign = -sign(squeeze(corrq_rho(dfsi,:))); corrq_rho_plt = corrq_sign.*squeeze(corrq_arsq(dfsi,:));
    corrq_sign(corrq_sign==1)=0; if dfsi == 2,corrq_sign(:) = 0; end; lowbounds = squeeze(corrq_r2Up(dfsi,:))-(corrq_rho_plt.*corrq_sign*2);
    upbounds = squeeze(corrq_r2Low(dfsi,:))-(corrq_rho_plt.*corrq_sign*2);
 
    plot(dfreqs,lowbounds,'Color', plcolormain(dfsi,:), 'LineWidth',0.5); % plot bounds
    hold on
    plot(dfreqs,upbounds,'Color', plcolormain(dfsi,:), 'LineWidth',0.5); % plot bounds
    plot(dfreqs,corrq_rho_plt,'Color', plcolormain(dfsi,:), 'LineWidth',2); % plot correlation coefficients across freqs   
    end
    
    if ~any(get(gca,'YLim')==0), plot(df0,zeros([1 2]),':k'); end
    ylabel('-sign(\beta^{2}) R ^{2}')
    meanactN = mean(squeeze(nonoutN(dfsi,:)));
    title(figfptitl,'interpreter','none')
    hax = gca;
    set(hax,'XLim',df0,'XTick',dfxticks,'XTickLabel',dfxticksl,'XMinorTick', 'on', 'XScale', 'log','FontSize',20,'FontName','Myriad Pro')
    set(hax.XAxis,'MinorTickValues', dfxminticks)
    xlabel('Frequency [Hz]')
    box off, hold off 
    
subplot(1,2,2)
    log10_corrq_pval = -log10(corrq_pval);
    plot(log10_corrq_pval,'.-','color',plcolorfaint)  
    hold on
    log10_corrq_fdr = -log10(corrq_fdr);
    plot(log10_corrq_fdr,'.-','color',plcolorfainter)
    
    log10_corrq_fdrb = -log10(corrq_fdrb);
    plot(log10_corrq_fdrb,'.-','Color',plcolormain(1,:)), hold on 
    plot(0:nfreqs+1,ones([1 nfreqs+2])*-log10(0.05),':k'); plot(0:nfreqs+1,ones([1 nfreqs+2])*-log10(0.01),':k'); plot(0:nfreqs+1,ones([1 nfreqs+2])*-log10(0.001),':k');
    
    ylims = get(gca,'YLim'); set(gca,'YLim',[0,ylims(2)]);
    ylabel([PerSpear ' p-value'])
    set(gca,'XLim',[0 nfreqs+1],'XTick',1:nfreqs,'XTickLabel',dfreqs)
    yticklabls = get(gca,'YTickLabel');  set(gca,'YTickLabel',['1';cellfun(@(x) ['10^{-' x '}'], yticklabls(2:end),'unif',0)])
    xlabel('Frequency [Hz]')
    for iptxt = 1:nfreqs
        postxt = log10_corrq_fdrb(iptxt)+ ylims(2)/40; if ylims(2)-postxt < ylims(2)/20, postxt = log10_corrq_fdrb(iptxt)-ylims(2)/40; end
        %if postxt == Inf, postxt = ylims(2)-ylims(2)/40; end %in case of pvals = 0
        text(iptxt,postxt,num2str(-log10_corrq_fdrb(iptxt),2),'HorizontalAlignment','center')
    end

    title(['pvals ' efidmaskstr ' ' sfidmaskstr])
    box off
 
print (figfpcorr,'-dpng', '-r400', [paths.output '_DFA_wPLI_' PerSpear 'rpval_pquadr.png']) %png
%     print (figfpcorr,'-dmeta','-painters', [paths.output '_DFA_wPLI_' PerSpear 'rpval_pquadr.emf']) %emf
%     close(figfpcorr)


%% Single frequency correlations plot

%% Figure 2 E F (Single regressions)
% of a single frequency across subjects 
freqreg = 10; % freq regression to plot in Hz (7 and 10 in papaer)
ffreqri = find(dfreqs==freqreg);

plcolormain = [0 .5 .8]; plcolorlight = [.7 .8 1]; plcolorfainter = [.8 .8 .8]; 
lincolorm =  [0.42 .66 .42]; quadcolorm = [0.93 0.73 0.33];  

plvsmeans_r = syncgsm(:,ffreqri);
dfdatams_r = dfdatams(:,ffreqri);

[plvsmeans_r, plvsmeans_routi] = rmoutliers(plvsmeans_r); dfdatams_r(plvsmeans_routi) = []; 
noutNf = numel(plvsmeans_r);
nonoutNf = num2str(noutNf);
                
lregsrts = regstats(dfdatams_r,plvsmeans_r,'linear');
qregsrts = regstats(lregsrts.r,plvsmeans_r,'quadratic');
% [corr_rhof, corr_pvalf] = corr(plvsmeans_r,dfdatams_r,'Type',PerSpear);

% linear
betal = lregsrts.beta;
corrl_pvalf= lregsrts.tstat.pval(2);

[X, ixd] = sort(plvsmeans_r);
Y = ones(size(X))*betal(1) + betal(2)*X; % regline
y = dfdatams_r(ixd);
% regression margin
alpha_int = 0.05;
SE_y_cond_x = sum((y - betal(1)*ones(size(y))-betal(2)*X).^2)/(noutNf-2);
SSX = (noutNf-1)*var(X);
SE_Y = SE_y_cond_x*(ones(size(X))*(1/noutNf + (mean(X)^2)/SSX) + (X.^2 - 2*mean(X)*X)/SSX);
Yoff = (2*finv(1-alpha_int,2,noutNf-2)*SE_Y).^0.5;
top_int = Y + Yoff;
bot_int = Y - Yoff;

% (partial) quadratic
betaq = qregsrts.beta;
corrq_pvalf= qregsrts.tstat.pval(2);

% Yq = ones(size(X))*beta(1) + beta(2)*X + beta(3)*(X.^2); % quadratic regline

% alpha=0.05 regression margin
% in the case of quadratic, we could get the margins numerically like in the linear version above,
% but using fit works well too and the fitted line corresponds to regstats' quadratic regline
[population2,gof] = fit(plvsmeans_r,dfdatams_r,'poly2');
% this is part of fitting toolbox, unlike functions above here no way around it
% in case comment line above and "plot quadratic fit + regr margins" below
% gof.adjrsquare == regsrts.adjrsquare

%% regression plot
figfrtitl = ['Fig2F Subjs Correlation DFA-' connmetric ' '  num2str(freqreg) 'Hz'];
    figfrcorr = figure ('Name',figfrtitl,'PaperPositionMode','auto','Position', fscrize2, 'Color', [1 1 1]); 
   subplot(121)
    
scatter(X,y,25,'MarkerFaceColor',plcolorlight,'MarkerEdgeColor',plcolormain);
hold on
plot(X,Y,'Color',lincolorm,'LineWidth',3);
plot(X,top_int,':', 'Color', lincolorm,'LineWidth',1);
plot(X,bot_int,':', 'Color', lincolorm,'LineWidth',1);

%% plot quadratic fit + regr margins
po2 = plot(population2,'predobs');
for ipc = 1:3, po2(ipc).Color = quadcolorm;  po2(ipc).LineWidth = 2; end
 po2(1).LineWidth = 3;

%% refine and save
title(['MEG ' connmetric '-DFA Correlation ' num2str(freqreg) 'Hz'],'interpreter','none');
ylabel('Mean DFA')
xlabel(['Mean Parcels ' connmetric],'interpreter','none') %GS
hax = gca;
    set(hax,'FontSize',20,'FontName','Myriad Pro')
    box off, hold off 
    
    
subplot(122)  %just for the annotation
hax = gca;
poss = hax.Position;
posann = [poss(1)+0.2,0.3,0.1,0.02];
posannq = [poss(1)+0.2,0.7,0.1,0.02];

annotation('textbox',posann ,'string',...
    ['lin rho : ' num2str(betal(2)) newline 'pval : ' num2str(corrl_pvalf)...
     newline 'Adjusted r^2 : ' num2str(lregsrts.adjrsquare) newline 'Adjusted pval : ' num2str(lregsrts.fstat.pval)],'LineStyle','none')
         
 annotation('textbox',posannq ,'string',...
    ['quad \beta x : ' num2str(betaq(2)) newline '; \beta x^2 : ' num2str(betaq(3)) newline 'pval : ' num2str(corrq_pvalf)...
     newline 'Adjusted r^2 : ' num2str(qregsrts.adjrsquare) newline 'Adjusted pval : ' num2str(qregsrts.fstat.pval)],'LineStyle','none')
 
title([figfrtitl ' ' efidmaskstr ' ' sfidmaskstr ' N = ' nonoutNf],'interpreter','none')

print (figfrcorr,'-dpng', '-r400', [paths.output '_DFA_' connmetric '_' num2str(freqreg) 'Hz_regrs.png']) %png
% close(figfrcorr)


%%


