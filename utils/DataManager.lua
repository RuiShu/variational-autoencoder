require 'hdf5'

local DataManager = torch.class("DataManager")

function DataManager:__init(batchsize)
   self.batchsize = batchsize
   self.indices = nil
   self.current = nil
   self.train = nil
end

function DataManager:inEpoch()
   return self.current ~= #self.indices
end

function DataManager:shuffle(batchsize)
   self.batchsize = batchsize or self.batchsize
   self.indices = torch.randperm(self.train:size(1)):long():split(self.batchsize)
   self.indices[#self.indices] = nil
   self.current = 0
end

function DataManager:next()
   self.current = self.current + 1
   xlua.progress(self.current, #self.indices)
   local v = self.indices[self.current]
   local inputs = self.train:index(1, v)
   return {inputs}
end

function DataManager:cuda()
   require 'cunn'
   self.train = self.train:cuda()
   return self
end
