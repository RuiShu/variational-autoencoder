require 'nn'

local Sampler, parent = torch.class('nn.Sampler', 'nn.Module')

function Sampler:__init()
   parent.__init(self)
   self.eps = torch.Tensor()
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
   self.gradInput:resizeAs(input)
   self.dMu, self.dLv = unpack(self.gradInput:split(self.nMix, self.len-1))
   self.dMu:copy(gradOutput)
   self.dLv:copy(self.lv):div(2):exp():cmul(self.eps):cmul(gradOutput)
   return self.gradInput
end

function Sampler:_viewInput(input)
   self.len = input:dim()
   self.nMix = input:size(self.len-1)/2
   self.mu, self.lv = unpack(input:split(self.nMix, self.len-1))
   self.mu = self.mu:squeeze(self.len-1)
   self.lv = self.lv:squeeze(self.len-1)
end
