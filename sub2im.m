function im = sub2im(x,y)
    sz = floor(size(x,1)*0.05);
    pd = floor(sz*0.2);
    sz = sz-pd;

    xd = abs(max(x) - min(x));
    yd = abs(max(y) - min(y));

    xinc = floor(sz*(1-(xd/yd*((xd/yd)<=1))-(1*((xd/yd)>1))));
    yinc = floor(sz*(1-(yd/xd*((yd/xd)<=1))-(1*((yd/xd)>1))));

    xxsz = sz-xinc;
    yysz = sz-yinc;

    xx = floor(rescale(x,1,xxsz));
    yy = floor(rescale(y,1,yysz));

    idx = sub2ind([yysz xxsz],yy,xx);

    im = reshape(hist(idx,1:yysz*xxsz),[yysz xxsz]);
    im = im/max(im(:));
    im = padarray(im,[pd pd],0);

    for i = 1:3
        im = imresize(imresize(im,2),0.5);
        im = im/max(im(:));
    end
end