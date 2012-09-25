function cmap = rand_cmap_alphb_size(alphabet_size)
   alphabet        = linspace(0, 255, alphabet_size);
   cmap            = allcomb(alphabet,alphabet,alphabet);

   %% Given by the alphabet size
   cmap            = uint8(cmap);     
   
   % Get rid of white:
   cmap(end,:)     = [];
   
   perm_indices    = randperm(size(cmap,1));
   black_index     = perm_indices == 1;

   cmap            = cmap(perm_indices,:);
   cmap(black_index,:) = cmap(1,:);
   cmap(1,:)           = [0 0 0];

end

