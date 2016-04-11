require 'hdf5'
require 'utils.DataManager'
local MnistManager, parent = torch.class('MnistManager', 'DataManager')

function MnistManager:__init(batchsize)
   parent.__init(self, batchsize)
   -- get data
   local f = hdf5.open('datasets/mnist.hdf5', 'r')
   self.train = f:read('x_train'):all():double()
   f:close()
end
