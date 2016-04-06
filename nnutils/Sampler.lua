-- Based on JoinTable module
require 'nn'

local Sampler, parent = torch.class('nn.Sampler', 'nn.Module')

function Sampler:__init()
   parent.__init(self)
   self.eps = torch.Tensor()
end 

function Sampler:updateOutput(input)
   -- input: batchsize x 2*nMix x nDims
   local eps = self.eps
   local output = self.output
   local len = input:dim()
   local nMix, mu, lv
   if len == 3 then
      nMix = input:size(2)/2
      mu = input[{{},{1,nMix},{}}]
      lv = input[{{},{nMix+1,2*nMix},{}}]
   elseif len == 2 then
      nMix = input:size(1)/2
      mu = input[{{1,nMix},{}}]
      lv = input[{{nMix+1,2*nMix},{}}]
   end
   if nMix == 1 then
      eps:resizeAs(lv):copy(torch.randn(lv:size()))
      output:resizeAs(lv):copy(lv)
      output:div(2):exp():cmul(eps):add(mu)
   end
   self.len = len
   self.nMix = nMix
   self.mu = mu
   self.lv = lv
   self.output = output:squeeze(len-1)
   return self.output
end

function Sampler:updateGradInput(input, gradOutput)
   local gradInput = self.gradInput
   local eps = self.eps
   local len = self.len
   local nMix = self.nMix
   local mu = self.mu
   local lv = self.lv
   gradInput:resizeAs(input)
   if len == 3 then
      dMu = gradInput[{{},{1,nMix},{}}]
      dLv = gradInput[{{},{nMix+1,2*nMix},{}}]
   elseif len == 2 then
      dMu = gradInput[{{1,nMix},{}}]
      dLv = gradInput[{{nMix+1,2*nMix},{}}]
   end
   dMu:copy(gradOutput)
   dLv:copy(lv):div(2):exp():cmul(eps):cmul(gradOutput)
   return self.gradInput
end
