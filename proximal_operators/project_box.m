function x = project_box(b,l,u)

% Project a point onto a box
% min_x ||x-b||_2, s.t., l <= x <= u
%
% version 1.0 - 18/06/2016
%
% Written by Canyi Lu (canyilu@gmail.com)
% 

x = max(l,min(b,u));