grid = {}

function grid.stack(images, nRow, nCol)
   -- nBatch x Height x Width
   local H = images:size(2)
   local W = images:size(3)
   local grid = torch.Tensor(nRow, nCol, H, W)
   local idx = 0
   for i = 1,nRow do
      for j = 1,nCol do
         idx = idx + 1
         grid[{i,j}]:copy(images[idx])
      end
   end
   -- indexing tricks
   grid = grid:transpose(3, 4):contiguous():view(nRow, nCol*W, H)
   grid = grid:transpose(2, 3):contiguous():view(nRow*H, nCol*W)
   return grid
end
