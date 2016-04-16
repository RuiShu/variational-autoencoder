require 'hdf5'
require 'utils.DataManager'
local FreyManager, parent = torch.class('FreyManager', 'DataManager')

function FreyManager:__init(batchsize)
   parent.__init(self, batchsize)
   -- get data
   local f = hdf5.open('datasets/freyfaces.hdf5', 'r')
   self.train = (1-f:read('train'):all():double()):csub(0.5):mul(2)
   f:close()
end

