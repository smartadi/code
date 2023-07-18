
%% Define the plant and model
Am=A; %The model with error
%C=[0.5 0 1]; % The output matrix
n=10
m=2
B=randn(n,m); % The input matrix
[rank(ctrb(Am,B)),rank(obsv(Am,C))] %check obsv and ctrb

%Lx=[-5]; % True error in the plant

%A=Am+B*Lx*C; %Formulate the true plant

%% Check for stable minimum phase transmission zeros (ASD Property)
P1=B*pinv(C*B)*C; 
P2=eye(n)-P1;
S2=null(C);
W2=S2';
A22=W2*P2*Am*W2';
eig(A22) % If two stable poles, system is ASD

%% Controller Synthesis

Fu=[0 0 0 0;0 1 0 0; 0 0 1 0;0 0 0 1]; %input generator

gamma_e=1*ones(size(C,1),1) %adaptivity parameter

Theta=[1 0 0 0;1 1 0 1]; %theta for input generator

%Bbar = padarray(B,[size(Fu,1),0],0,'post'); % Composite input matrix
Bbar = [B;zeros(size(Fu,1),size(B,2))] % Composite input matrix

Abar=[Am B*Theta; zeros(size(Fu,1),size(Am,2)) Fu]; %composite state matrix
eigAbar=eig(Abar); %check stability of composite matrix
Cbar=padarray(C,[0,size(Fu,1)],0,'post'); %composite output matrix

Q1=0.001*eye(length(Am)); %lqr Q matrix for state
Q2=0.005*eye(length(Fu)); %lqr Q matrix for input
[Klqr,Slqr,elqr]=dlqr(Abar',Cbar',blkdiag(Q1,Q2),95,0); %determine optimal observer gains
Klqr=Klqr.';
Kx=Klqr(1:length(Am),:);  %state estimator gains
Ku=Klqr(length(Am)+1:end,:); %input estimator gains
Ac_bar=Abar-Klqr*Cbar; %composite stabalized system
eig(Ac_bar) 



g_step=3; %step gain
g1=2; %sin(2t) gain
g2=0; %cos(2t) gain
g3=0; %sin(4t) gain
g4=4; %cos(4t) gain
g_missing=0; %missing gain






