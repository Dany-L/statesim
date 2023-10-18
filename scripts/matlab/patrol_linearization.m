clear all, close all
% parameters

%Vessel Data
% struct const.
% UPDATE: modified water density from 1014 to 1025, which is density of
% seawater
const.rho_water     =	1025.0;	        %	seawater density	[kg/m^3]
const.g				=	9.81;	        %	gravity constant	[m/s^2]			

% Main Particulars (Modified by T.Perez)
h.Lpp    =  51.5 ;                  %Length between perpendiculars [m]
h.B      =  8.6  ;                  %Beam over all  [m]
h.D	     =  2.3  ;                  %Draught [m]     

%Load condition (Modified by T.Perez) (Modified by A. Baier)
h.disp   =  355.88;                   %Displacement  [m^3]
h.m      =  365.79*10^3;  %Mass [Kg]
h.Izz    =  3.3818*10^7 ;            %Yaw Inertia
h.Ixx    =  3.4263*10^6;            %Roll Inertia [MODIFIED]
h.gm 		=  1.0;	                 %  [m]	Transverse Metacenter Height
h.LCG       = 20.41 ;                % [m]	Longitudinal CG (from AP considered at the rudder stock)
h.VCG       = 3.36  ;                % [m]	Vertical  CG  above baseline
h.xG        = -3.38  ;               % coordinate of CG from the body fixed frame adopted for the PMM test  
h.zG  	    = -1.06;          % coordinate of CG from the body fixed frame adopted for the PMM test  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The hydrodynamic derivatives are given in dimensional form, and follow
% from the original publication of Blanke and Christensen 1993.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data for surge equation 
h.Xudot  	= -17400.0 ;
h.Xuau     	= -1960.0 ;
h.Xvr    	=  0.33 * h.m ;
   
% Hydrodynamic coefficients in sway equation
h.Yvdot = -1.9022*10^6; 
h.Ypdot = -0.296*10^6; 
h.Yrdot = -1.4*10^6; 
h.Yauv  = -11800 ; 
h.Yur   =  131000 ; 
h.Yvav  = -3700 ; 
h.Yrar   =  0 ;
h.Yvar  = -794000 ; 
h.Yrav  = -182000 ; 
h.Ybauv =  10800 ; % Y_{\phi |v u|}
h.Ybaur =  251000 ; 
h.Ybuu  = -74 ; 

% Hydrodynamic coefficients in roll equation
h.Kvdot = 296000 ;
h.Kpdot = -674000 ;
h.Krdot =  0 ;
h.Kauv  = 9260 ;
h.Kur   = -102000 ;
h.Kvav  =  29300 ;
h.Krar  =  0 ;
h.Kvar  = 621000 ;
h.Krav  =  142000 ;
h.Kbauv =  -8400 ;
h.Kbaur =  -196000 ;
h.Kbuu  =  -1180 ;
h.Kaup  =  -15500 ;
h.Kpap  =  -416000 ;
h.Kp    = -500000;
h.Kb    =  0; %0.776*h.m*const.g;  % this coefficient is defined but missing in the original equation
h.Kbbb  =  -0.325*const.rho_water*const.g*h.disp;

% Hydrodynamic coefficients in yaw equation*)
h.Nvdot =  538000 ;
h.Npdot =  0 ;
h.Nrdot = -4.3928*10^7;
h.Nauv  = -92000 ;
h.Naur  = -4710000 ;
h.Nvav  =  0 ;
h.Nrar  = -202000000 ; % TODO: this coefficient is extremely large
h.Nvar  =  0 ;
h.Nrav  = -15600000 ;
h.Nbauv = -214000 ;
h.Nbuar = -4980000 ;
h.Nbuau = -8000 ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Total Mass Matrix 
M=[(h.m-h.Xudot)  0   0   0   0;
   0 (h.m-h.Yvdot) -(h.m*h.zG+h.Ypdot) (h.m*h.xG-h.Yrdot) 0 ;
   0 -(h.m*h.zG+h.Kvdot) (h.Ixx-h.Kpdot) -h.Krdot 0 ;
   0 (h.m*h.xG-h.Nvdot) -h.Npdot (h.Izz-h.Nrdot) 0 ;
   0 0 0 0 1 ];

N = [10*h.Xuau 0 0 0 0;
    0 5*h.Yauv 0 5*h.Yur-5*h.m 25*h.Ybauv;
    0 (5*h.Kauv) (5*h.Kaup+h.Kp) (5*h.Kur + 5*h.m*h.zG) (h.Kb + 25*h.Kbuu -const.rho_water*const.g*h.disp*h.gm);
    0 5*h.Nauv 0 (5*h.Naur-5*h.m*h.xG) (25*h.Nbuau);
    0 0 1 0 0];

T = [1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1;
    0 0 0 0];
A_ship_c = M^(-1)*N;
B_ship_c = M^(-1)*T;