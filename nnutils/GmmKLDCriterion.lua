require 'nn'

local GmmKLDCriterion, parent = torch.class('nn.GmmKLDCriterion', 'nn.Criterion')

function GmmKLDCriterion:__init()
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   -- create buffers
   self.lvDiff = torch.Tensor()
   self.qExp = torch.Tensor()
   self.muDiff = torch.Tensor()
   self.expDiff = torch.Tensor()
   self.expElem = torch.Tensor()
   self.expSum = torch.Tensor()
   self.preOut = torch.Tensor()
   self.dpMuBuf = torch.Tensor()
   self.dpLvBuf = torch.Tensor()
   self.preKld = torch.Tensor()
   self.shift = torch.Tensor()
end 

function GmmKLDCriterion:updateOutput(p, q)
   -- p is batchSize x 2 x nDims
   -- q is batchSize x (2*nMixtures) x nDims
   self:_viewInput(p, q)
   self:_resizeBuffers(p, q)
   self.pMu = self.pMu:expandAs(self.qMu)
   self.pLv = self.pLv:expandAs(self.qLv)
   self.lvDiff:add(self.pLv, -1, self.qLv)
   self.muDiff:add(self.pMu, -1, self.qMu)
   self.expDiff:exp(self.lvDiff)
   self.qExp:mul(self.qLv, -1):exp()
   self.preKld:pow(self.muDiff, 2):cmul(self.qExp):add(self.expDiff):csub(1):csub(self.lvDiff)

   -- prevent overflow/underflow
   self.expElem:sum(self.preKld, self.len):div(2)
   self.shift:min(self.expElem, self.len-1)
   self.shiftEx = self.shift:expandAs(self.expElem)
   self.expElem:neg():add(self.shiftEx):exp()
   self.expSum:sum(self.expElem, self.len-1)
   self.preOut:div(self.expSum, self.nMix):log():neg():add(self.shift)
   self.output = self.preOut:sum()
   return self.output
end

function GmmKLDCriterion:updateGradInput(p, q)
   self.expSum = self.expSum:expandAs(self.expElem)
   self.expElem:cdiv(self.expSum)
   self.expElem = self.expElem:expandAs(self.qExp)
   -- compute dp
   self.dpMuBuf:cmul(self.muDiff, self.qExp):cmul(self.expElem)
   self.dpLvBuf:csub(self.expDiff, 1):div(2):cmul(self.expElem)
   self.dpMu:sum(self.dpMuBuf, self.len-1)
   self.dpLv:sum(self.dpLvBuf, self.len-1)
   -- compute dq
   self.dqMu:mul(self.muDiff, -1):cmul(self.qExp):cmul(self.expElem)
   self.dqLv:pow(self.muDiff, 2):cmul(self.qExp):neg():csub(self.expDiff):add(1):div(2):cmul(self.expElem)
   return self.gradInput[1], self.gradInput[2]
end

function GmmKLDCriterion:_resizeBuffers(p, q)
   self.lvDiff:resizeAs(self.qLv)
   self.qExp:resizeAs(self.qLv)
   self.muDiff:resizeAs(self.qLv)
   self.expDiff:resizeAs(self.qLv)
   self.preKld:resizeAs(self.qLv)
   local size = self.qLv:size()
   size[#size] = 1
   self.expElem:resize(size)
   size[#size-1] = 1
   self.expSum:resize(size)
   self.preOut:resize(size)
   self.shift:resize(size)
   self.dpMuBuf:resizeAs(self.qLv)
   self.dpLvBuf:resizeAs(self.qLv)
end

function GmmKLDCriterion:_viewInput(p, q)
   self.len = q:dim()
   self.nMix = q:size(self.len-1)/2
   self.pMu, self.pLv = unpack(p:split(1, self.len-1))
   self.qMu, self.qLv = unpack(q:split(self.nMix, self.len-1))
   -- grad view
   self.gradInput[1]:resizeAs(p)
   self.gradInput[2]:resizeAs(q)
   self.dpMu, self.dpLv = unpack(self.gradInput[1]:split(1, self.len-1))
   self.dqMu, self.dqLv = unpack(self.gradInput[2]:split(self.nMix, self.len-1))
end   
