function [outimg] = integral( image )
[y,x] = size(image);
outimg = zeros(y+1,x+1);
disp(y);
for a = 1:y+1
    for  b = 1:x+1
        rx = b-1;
        ry = a-1;
        while ry>=1
            while rx>=1  
                outimg(a,b) = outimg(a,b)+image(ry,rx);
                rx = rx-1;
            end
            rx = b-1;
            ry = ry-1;
        end
        % outimg(a,b) = outimg(a,b)-image(a,b);
    end   
end
% outimg(1,1) = image(1,1);
disp('end loop');
end