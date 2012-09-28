function u_vals_diff = faster_unique(lbl_array)
   u_vals_diff = [];
   if ~isempty(lbl_array)
         lbl_array = sort(lbl_array);
         d = diff(lbl_array);
         d = d ~=0;
         d = [true; d];
         u_vals_diff = lbl_array(d);   
   end  
   u_vals_diff = cast(u_vals_diff, class(lbl_array));
end