require 'hdf5'

local MnistManager = torch.class("MnistManager")

function MnistManager:__init(batchsize)
   self.batchsize = batchsize
   -- get data
   local f = hdf5.open('datasets/mnist.hdf5', 'r')
   self.train = f:read('x_train'):all():double()
   f:close()
   self.indices = nil
   self.current = nil
end

function MnistManager:inEpoch()
   return self.current ~= #self.indices
end

function MnistManager:shuffle(batchsize)
   self.batchsize = batchsize or self.batchsize
   self.indices = torch.randperm(self.train:size(1)):long():split(self.batchsize)
   self.indices[#self.indices] = nil
   self.current = 0
end

function MnistManager:next()
   self.current = self.current + 1
   xlua.progress(self.current, #self.indices)
   local v = self.indices[self.current]
   local inputs = self.train:index(1, v)
   return {inputs}
end

function MnistManager:cuda()
   require 'cunn'
   self.train = self.train:cuda()
   return self
end
