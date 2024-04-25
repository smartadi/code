clear all;
close all;
clc;
% dt = 1/35;
dt = 0.0285;
t = 0:dt:dt*1000;


path = "/run/user/1001/gvfs/smb-share:server=sahale.biostr.washington.edu,share=data/Subjects/AB_0032/2024-03-14/";
param_path = '2/2024-03-14_2_AB_0032_Block.mat';
param_path = append(path,param_path);
i_path = '1/lightCommand.raw.npy'
i_path = append(path,i_path);
input = readUfromNPY(i_path);
a= load(param_path)
tr_path = '1/lightCommand.timestamps_Timeline.npy';
tr_path = append(path,tr_path);
tr = readUfromNPY(tr_path);

exp_path = '1/expStartStop.raw.npy';
expt_path = '1/expStartStop.timestamps_Timeline.npy';


exp_path = append(path,exp_path);
expt_path = append(path,expt_path);

exp = readVfromNPY(exp_path);
expt = readVfromNPY(expt_path);



vpath = '1/corr/svdTemporalComponents_corr.npy';
vpath = append(path,vpath);

vtpath = '1/corr/svdTemporalComponents_corr.timestamps.npy';
vtpath = append(path,vtpath);




Vv_raw = readVfromNPY(vpath);
Vt = readVfromNPY(vtpath);


Vv1 = Vv_raw./vecnorm(Vv_raw,2,1);

tpath = '1/frameTimes.timestamps.npy' 
tpath = append(path,tpath);

tt = readVfromNPY(tpath);
%% filter



l=10;
S = Vv_raw';
%
close all;
fc =6;
n = 8;
fs = 35;
%[bb,aa] = butter(6,fc/(fs/2));

[bb,aa] = butter(2,fc/(fs/2));
y = filter(bb,aa,S);



Vv = (y./vecnorm(y,2,2))';
%%
figure()
plot(Vv(1:10,1:1000)')

figure()
plot(Vv1(1:10,1:1000)')


%% sort the inputs

te = a.block.events;
tp = a.block.paramsValues;
tpt = a.block.paramsTimes;

frames = tr(2,1) - tr(1,1);

total_time = tr(2,2) - tr(1,2);

t_series = tr(1,2):total_time/frames:tr(2,2);
events = vertcat(tp.trialType);

%%
figure()
plot(t_series,input)
%%

vtp = Vt(1:end-1);
vtm = Vt(2:end);

Vdt = vtm-vtp;

%
figure()
plot(Vdt(1:100))


%% input interp

input_interp = interp1(t_series,input,Vt);


%%
figure()
plot(Vt(40000:42000),input_interp(40000:42000))

%% Sort data

freq_startTimes = [];
freq_endTimes = [];
%%
close all;
figure()
plot(Vt(1:19000),input_interp(1:19000))


figure()
plot(Vt(15000:20000),input_interp(15000:20000))
%

figure()
plot(Vt(15000:20000),Vv(2,15000:20000))


figure()
plot(Vt(1:1000),Vv(2,1:1000))
%%
% wf_interp = interp1(input_interp,input,Vt);

%%
close all;

figure()
plot(t_series(1075000:1150000),input(1075000:1150000))

figure()
plot(t_series(8075000:end),input(8075000:end))


figure()
plot(Vt(15000:30000),input_interp(15000:30000))


%%
figure()
plot(t_series(1:3000000),exp(1:3000000))

%%
close all

st = a.block.events.trialNumTimes;
en = a.block.events.endTrialTimes;


s1 = st(1:end-1);
s2 = st(2:end);

e1 = en(1:end-1);
e2 = en(2:end);



ss = s2-s1;
ee = e2-e1;
se = en-st;

figure()
subplot(3,1,1)
plot(ss)
subplot(3,1,2)
plot(ee)
subplot(3,1,3)
plot(se)

%% start sorting
close all;

V_e = vecnorm(Vv_raw,2,1);

figure()
subplot(3,1,1)
plot(Vt(18000:20000),Vv(1:5,18000:20000));
subplot(3,1,2)
plot(Vt(18000:20000),input_interp(18000:20000));
subplot(3,1,3)
plot(Vt(18000:20000),V_e(18000:20000));


figure()
subplot(3,1,1)
plot(Vt(15000:30000),Vv(1:5,15000:30000));
subplot(3,1,2)
plot(Vt(15000:30000),input_interp(15000:30000))
subplot(3,1,3)
plot(Vt(15000:30000),V_e(15000:30000))


figure()
subplot(3,1,1)
plot(Vt(18654:19800),Vv(1:5,18654:19800));
ylabel('WF signal')
subplot(3,1,2)
plot(Vt(18654:19800),input_interp(18654:19800));
ylabel('Inputs')
subplot(3,1,3)
plot(Vt(18654:19800),V_e(18654:19800))
ylabel('energy')
xlabel('time')

figure()
subplot(3,1,1)
plot(Vt(18654:19800),Vv_raw(1:5,18654:19800));
ylabel('WF signal')
subplot(3,1,2)
plot(Vt(18654:19800),input_interp(18654:19800))
ylabel('Inputs')
subplot(3,1,3)
plot(Vt(18654:19800),V_e(18654:19800))
ylabel('energy')
xlabel('time')

%%
figure()
subplot(3,1,1)
plot(Vt(15000:19800),Vv_raw(1:5,15000:19800));
ylabel('WF signal')
subplot(3,1,2)
plot(Vt(15000:19800),input_interp(15000:19800))
ylabel('Inputs')
subplot(3,1,3)
plot(Vt(15000:19800),V_e(15000:19800))
ylabel('energy')
%%

figure()
plot(Vt(1:10000),V_e(1:10000))


figure()
plot(Vt(1:100000),V_e(1:100000))


%%
load sdata2.mat umat2 ymat2

np = 2;
nz = 0;
iodelay = 2;
Ts = 0.1;
sysd = tfest(umat2,ymat2,np,nz,iodelay,'Ts',Ts)
%%
% x = input_interp(18654:19800)';
% y = Vv(1,18654:19800)';

x = input_interp(18654:19000)';
y = double(Vv(2,18654:19000))';

txy = tfestimate(x,y);
%%

np = 10;
nz = 5;
sysd = tfest(x,y,np,nz,'Ts',1/35)

%%
np=5;
nz=0;
sysd = tfest(x,y,np,nz,'Ts',1/35)
%
close all;
bode(sysd)

%% sort
events = vertcat(tp.trialType);

off_index = 18650;
offset = Vt(off_index);
Vf=[];
Vfr=[];
Vi=[];
Vir=[];
tf=[];
ti=[];
inpf=[];
inpi=[];
% fl = 1225;
fl = 1100;
% il = 53;
il = 40;
for i=1:length(events)
    [ds,idx_start] = min(abs(Vt - (offset+st(i))));
    [de,idx_end] = min(abs(Vt - (offset+en(i))));
    
    gap = idx_end-idx_start
    if events(i) == 0
        % Vf = [Vf; Vv(idx_start:idx_end)];
        Vf = [Vf; Vv(1:10,idx_start:idx_start+fl)];
        Vfr = [Vfr; Vv_raw(1:10,idx_start:idx_start+fl)];
        inpf = [inpf; input_interp(idx_start:idx_start+fl)];
        tf = [tf; Vt(idx_start:idx_start+fl)];
    else
        % Vi = [Vi; Vv(idx_start:idx_end)];
        Vi = [Vi; Vv(1:10,idx_start:idx_start+il)];
        Vir = [Vir; Vv_raw(1:10,idx_start:idx_start+il)];
        inpi = [inpi; input_interp(idx_start:idx_start+il)];
        ti = [ti; Vt(idx_start:idx_start+il)];
    end

end




%
close all;

figure()
plot(ti(1,:),inpi);


figure()
plot(tf(1,:),inpf);


figure()
plot(tf(1,:),Vf);

figure()
plot(ti(1,:),Vi);



% figure()
% plot(Vt(off_index:off_index+1225),input_interp(off_index:off_index+1225))


%%  save input output data
% writematrix(Vfr,path+'1/input/fs_output.csv');
% writematrix(inpf,path+'1/input/fs_input.csv');
% writematrix(tf,path+'1/input/fs_time.csv');
% 
% writematrix(Vir,path+'1/input/imp_output.csv');
% writematrix(inpi,path+'1/input/imp_input.csv');
% writematrix(ti,path+'1/input/imp_time.csv');
