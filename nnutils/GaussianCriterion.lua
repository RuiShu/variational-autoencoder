require 'nn'

local GaussianCriterion, parent = torch.class('nn.GaussianCriterion', 'nn.Criterion')

function GaussianCriterion:__init()
   parent.__init(self)
   self.diff = torch.Tensor()
   self.pExp = torch.Tensor()
   self.diffExp = torch.Tensor()
   self.expElem = torch.Tensor()
end

function GaussianCriterion:updateOutput(p, x)
   -- negative log likelihood using p as gaussian distribution hypothesis
   -- and x as observation
   self:_viewInput(p, x)
   self:_resizeBuffers()
   self.diff:csub(self.x, self.pMu)
   self.pExp:mul(self.pLv, -1):exp()
   self.diffExp:pow(self.diff, 2):cmul(self.pExp)
   self.expElem:add(self.diffExp, self.pLv):add(math.log(2*math.pi))
   self.output = self.expElem:sum()*0.5
   return self.output 
end

function GaussianCriterion:updateGradInput(p, x)
   self.dpMu:cmul(self.diff, self.pExp):neg()
   self.dpLv:csub(self.diffExp, 1):div(2):neg()
   return self.gradInput
end

function GaussianCriterion:_resizeBuffers()
   self.diff:resizeAs(self.pMu)
   self.pExp:resizeAs(self.pMu)
   self.diffExp:resizeAs(self.pMu)
   self.expElem:resizeAs(self.pMu)
end

function GaussianCriterion:_viewInput(p, x)
   -- nBatch x nChannel*2 x nDim
   self.len = p:dim()
   self.nChannel = p:size(self.len-1)/2
   self.pMu, self.pLv = unpack(p:split(self.nChannel, self.len-1))
   self.x = x:viewAs(self.pMu)
   -- grad view
   self.gradInput:resizeAs(p)
   self.dpMu, self.dpLv = unpack(self.gradInput:split(self.nChannel, self.len-1))
end
