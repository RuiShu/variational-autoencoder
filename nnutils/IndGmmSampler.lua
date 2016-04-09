require 'nn'

local Sampler, parent = torch.class('nn.IndGmmSampler', 'nn.Module')

function Sampler:__init()
   parent.__init(self)
   self.eps = torch.Tensor()
   self.muBuf = torch.Tensor()
   self.lvBuf = torch.Tensor()
   self.prob = torch.Tensor()
   self.idx = torch.LongTensor()
   self.lv = torch.Tensor()
   self.mu = torch.Tensor()
end 

function Sampler:updateOutput(input)
   -- input: batchsize x 2*nMix x nDims
   self:_viewInput(input)
   self.eps:resizeAs(self.lv):copy(torch.randn(self.lv:size()))
   self.output:resizeAs(self.lv):copy(self.lv)
   self.output:div(2):exp():cmul(self.eps):add(self.mu)
   return self.output
end

function Sampler:updateGradInput(input, gradOutput)
   -- skipping this for now.
   return self.gradInput
end

function Sampler:_viewInput(input)
   self.len = input:dim()
   self.nMix = input:size(self.len-1)/2
   self.muOrig, self.lvOrig = unpack(input:split(self.nMix, self.len-1))
   if self.len == 3 then
      -- crazy indexing tricks to perform gaussian mixture component
      -- sampling without for-loops
      local nBatch = input:size(1)
      local nDim = input:size(3)
      local nBatchDim = nBatch*nDim
      local muBufnonContig = self.muOrig:transpose(2,3)
      local lvBufnonContig = self.lvOrig:transpose(2,3)
      self.muBuf:resizeAs(muBufnonContig):copy(muBufnonContig)
      self.lvBuf:resizeAs(lvBufnonContig):copy(lvBufnonContig)
      self.muBuf = self.muBuf:view(nBatchDim, self.nMix)
      self.lvBuf = self.lvBuf:view(nBatchDim, self.nMix)
      -- sample components
      self.prob:resizeAs(self.lvBuf):fill(1/self.nMix)
      self.idx:resize(self.prob:size(1), 1)
      self.prob.multinomial(self.idx, self.prob, 1)
      self.mu:resize(nBatchDim,1)
      self.lv:resize(nBatchDim,1)
      self.mu:gather(self.muBuf, 2, self.idx)
      self.lv:gather(self.lvBuf, 2, self.idx)
      self.mu = self.mu:view(nBatch, nDim)
      self.lv = self.lv:view(nBatch, nDim)
   end
end
