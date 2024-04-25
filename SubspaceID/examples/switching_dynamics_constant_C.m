clear all;
close all;
clc;
% dt = 1/35;
dt = 0.0285;
t = 0:dt:dt*1000;

path = "/run/user/1001/gvfs/smb-share:server=steinmetzsuper1.biostr.washington.edu,share=data/Subjects/ZYE_0069/2023-10-03/1";
upath = '/corr/svdSpatialComponents_ortho.npy';
upath = append(path,upath);
vpath = '/corr/svdTemporalComponents_ortho.npy';
vpath = append(path,vpath);


% Uu = readUfromNPY(upath);
% Vv = readVfromNPY(vpath);
% 
% [TT,dims]= size(Vv);
% 
% Vv = Vv./vecnorm(Vv,2,2);
% 
% disp('loading WF PCA projections')
% 
% data1=Vv';
% N=10000;
% 
% l=10;
% S = Vv1(1:10000,1:l);


Uu = readUfromNPY(upath);
Vv1 = readVfromNPY(vpath);

[TT,dims]= size(Vv1);

Vv = Vv1./vecnorm(Vv1,2,2);

disp('loading WF PCA projections')

data1=Vv';
%%
N=20000;

l=10;
S = Vv1(1:20000,1:l);
%
close all;
fc = 10;
n = 8;
fs = 35;
[b,a] = butter(6,fc/(fs/2));
y = filter(b,a,S);



yn = (y./vecnorm(y,2,2))';
%%
% figure()
% plot(yn')
%%
n = 5;
l = n;
Ts = 1000;
data_small = yn(1:l,1:Ts);
nn = 4

%   We will now identify this system from the data y 
%   with the subspace identification algorithm: subid
%   
%   The only extra information we need is the "number of block rows" i
%   in the block Hankel matrices.  This number is easily determined
%   as follows:

%   Say we don't know the order, but think it is maximally equal to 10.
%   
       max_order = 20;
%   
%   As described in the help of subid we can determine "i" as follows:
%   
%       i = 2*(max_order)/(number of outputs)
%       must be an integer
       p = 2*(max_order)/l;

       p=25
  % nn= 10;

  AUX=[];

 [A,du1,C,du2,K,R,AUX] = subid(data_small,[],p,nn);
AUX=[];
 [A,B,C,D,K,Ro,AUX,ss,res,lhs,rhs] = subid_data(data_small,[],p,nn,AUX,'sv');


%%
dd = eig(A)
figure()
plot(dd,'ob');hold on

legend('unstable','stable')




%%

Q = TT/Ts;
% Q = 5.5;
Vd = zeros(nn,nn,floor(Q));
Ds=[];
D=[];
AA=[];
CC=[];

AAs=[];
CCs=[];
LHS = [];
RHS = [];

Xx=[];
Y=[];
X=[];
for i = 1:1:Q
    data = data1(1:l,Ts*(i-1)+1 : i*Ts);
    % data = data1(2:l,Ts*(i-1)+1 : i*Ts);

    AUX=[];
    [A,B,C,D,K,Ro,AUX,ss,res,Lhs,Rhs] = subid_data(data,[],p,nn,[],[],1);

    %[As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data,[],p,nn,AUX,'sv');

    %[Ad,Bd,Cd,Dd,Kd,Ro,AUX,ss,res,Lhs,Rhs] = subid_stable_data(data,[],p,nn,AUX,'sv');
    
    Xx = [Xx,Lhs(1:nn,:)];
    Y = [Y,Lhs(nn+1:end,:)];
    X = [X,Rhs];

    CC = [CC;C];
    AA = [AA;A];

    % if i < 11
    LHS = [LHS;Lhs];
    RHS = [RHS;Rhs];
    % end

    %As = As - Ks*Cs;RHS

    [E,V] = eig(A);
    
%    Vd(:,:,i) = V;

    D = [D,eig(A)];
    % Ds = [Ds,eig(As)];
% eig(A)
end
% %%
% close all;
% figure()
% plot(Ds,'or'); hold on;
% axis([-1 1 -1 1])
% title('stable eigenvalues')
% cDs = Ds/dt;
% cD = D/dt;
% 
% t= 1:1:Q;
% 
% figure()
% plot(t,abs(imag(cDs)),'or'); hold on;
% title('Stable continuous time eigenvalues')
% 
% figure()
% plot(t,abs(imag(cD)),'or'); hold on;
% title('Stable continuous')

%% Regression on C
C_constant = Y/X;

%% Change in system
eA=[];
eAs=[];

eC=[];
eCs=[];

t= 1:1:Q-1;

for i = 1:1:Q-1
    eA  = [eA; norm(AA(nn*(i-1)+ 1: nn*i,:) - AA(nn*(i)+ 1: nn*(i+1),:))];
    % eAs = [eAs; norm(AAs(nn*(i-1)+ 1: nn*i,:) - AAs(nn*(i)+ 1: nn*(i+1),:))];

    eC  = [eC; norm(CC(l*(i-1)+ 1: l*i,:) - CC(l*(i)+ 1: l*(i+1),:))];
    % eCs = [eCs; norm(CCs(l*(i-1)+ 1: l*i,:) - CCs(l*(i)+ 1: l*(i+1),:))];

end

figure()
plot(t,eA); hold on;

title("Fro norm of change in A")

figure()
plot(t,eC); hold on;

title("Fro norm of change in C")



%% Switching data

% [As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data_small,[],p,nn,AUX,'sv');
% 
% [Ad,Bd,Cd,Dd,Kd,Ro,AUX,ss,res,Lhs,Rhs] = subid_stable_data(data_small,[],p,nn,AUX,'sv');

C = C_constant
T=100;
cvx_begin
    delta = 1e-2

    variable Th((nn)*T,nn)
    % II = eye(T)
    % TTh = kron(Th,II);
    for i=1:T
        % size(Th((i-1)*(nn)+1:(i)*(nn),:))
        % size(C)
        obj = sum(norm([Th((i-1)*(nn)+1:(i)*(nn),:);C]*RHS((i-1)*nn+1:i*nn,:)-LHS((i-1)*(nn+l)+1:i*(nn+l),:)));
    end
    
    minimize( obj )
    subject to
        for i = 1:T-1
            norm(Th((i-1)*(nn)+1:(i-1)*(nn)+nn,:) - Th((i)*(nn)+1:(i)*(nn)+nn,:)) <= delta
        end
            

cvx_end


% props of A
D=[];
e=[];
for i=1:T-1
    A = Th((i-1)*(nn)+1:(i-1)*(nn)+nn,:);

    A2 = Th((i)*(nn)+1:(i)*(nn)+nn,:);
  
    A = logm(A)/dt;
    [E,V] = eig(A);

    D = [D,eig(A)];
    e=[e;norm(A-A2)];


end


%

close all;
figure()
plot(e)

t= 1:1:T-1;

figure()
plot(t,abs(imag(D)),'ob'); hold on;
title('learnt freq')


figure()
plot(D,'or'); hold on;
axis([-1 1 -1 1])
title('eigenvalues')


%% Switching data 2

% [As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data_small,[],p,nn,AUX,'sv');
% 
% [Ad,Bd,Cd,Dd,Kd,Ro,AUX,ss,res,Lhs,Rhs] = subid_stable_data(data_small,[],p,nn,AUX,'sv');

C = C_constant
T=20;
cvx_begin
    delta = 1e-2

    variable Th((nn)*T,nn)
    II = eye(T)
    TTh = kron(Th,II);
    for i=1:T
        % size(Th((i-1)*(nn)+1:(i)*(nn),:))
        % size(C)
        obj = sum(norm([Th((i-1)*(nn)+1:(i)*(nn),:);C]*RHS((i-1)*nn+1:i*nn,:)-LHS((i-1)*(nn+l)+1:i*(nn+l),:)));
    end
    obj = norm(Xx(:,1:T) - TTh*RHS)
    
    minimize( obj )
    subject to
        for i = 1:T-1
            norm(Th((i-1)*(nn)+1:(i-1)*(nn)+nn) - Th((i)*(nn)+1:(i)*(nn)+nn)) <= delta
        end
            

cvx_end

%
figure()
plot(Lhs')