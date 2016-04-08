require 'nn'

local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

function KLDCriterion:__init()
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
   self.dpMu = torch.Tensor()
   self.dpLv = torch.Tensor()
end 

function KLDCriterion:updateOutput(p, q)
   -- p is batchSize x 2 x nDims
   -- q is batchSize x (2*nMixtures) x nDims
   self:_viewInput(p, q)
   self:_resizeBuffers(p, q)
   self.pMu = self.pMu:expandAs(self.qMu)
   self.pLv = self.pLv:expandAs(self.qLv)
   self.lvDiff:add(self.qLv, -1, self.pLv)
   self.muDiff:add(self.qMu, -1, self.pMu)
   self.expDiff:mul(self.lvDiff, -1):exp()
   self.qExp:mul(self.qLv, -1):exp()
   self.expElem:pow(self.muDiff, 2):cmul(self.qExp):add(self.expDiff):csub(1):add(self.lvDiff):div(2):neg():exp()
   self.expSum:sum(self.expElem, self.len-1)
   self.preOut:div(self.expSum, self.nMix):log():neg():sum()
   self.output = self.preOut:sum()
   return self.output
end

function KLDCriterion:updateGradInput(p, q)
   self.expSum = self.expSum:expandAs(self.expElem)
   self.expElem:cdiv(self.expSum)
   self.dpMu:copy(self.muDiff):cmul(self.qExp):neg():cmul(self.expElem)
   self.dpLv:copy(self.expDiff):csub(1):div(2):cmul(self.expElem)
   self.dqMu:copy(self.muDiff):cmul(self.qExp):cmul(self.expElem)
   self.dqLv:copy(self.muDiff):pow(2):cmul(self.qExp):neg():csub(self.expDiff):add(1):div(2):cmul(self.expElem)
   if self.len == 3 then
      self.gradInput[1][{{},{1},{}}]:sum(self.dpMu, 2)
      self.gradInput[1][{{},{2},{}}]:sum(self.dpLv, 2)
   elseif self.len == 2 then
      self.gradInput[1][{1,{}}]:sum(self.dpMu, 1)
      self.gradInput[1][{2,{}}]:sum(self.dpLv, 1)
   end
   return self.gradInput[1], self.gradInput[2]
end

function KLDCriterion:_resizeBuffers(p, q)
   -- cache from forward
   self.lvDiff:resizeAs(self.qLv)
   self.qExp:resizeAs(self.qLv)
   self.muDiff:resizeAs(self.qLv)
   self.expDiff:resizeAs(self.qLv)
   self.expElem:resizeAs(self.qLv)
   self.expSum:resizeAs(self.pLv)
   self.preOut:resizeAs(self.pLv)
   -- cache for backward
   self.dpMu:resizeAs(self.qLv)
   self.dpLv:resizeAs(self.qLv)
   -- grad storage
   self.gradInput[1]:resizeAs(p)
   self.gradInput[2]:resizeAs(q)
   if self.len == 3 then
      self.dqMu = self.gradInput[2][{{},{1,self.nMix},{}}]
      self.dqLv = self.gradInput[2][{{},{self.nMix+1,2*self.nMix},{}}]
   elseif self.len == 2 then
      self.dqMu = self.gradInput[2][{{1,self.nMix},{}}]
      self.dqLv = self.gradInput[2][{{self.nMix+1,2*self.nMix},{}}]
   end
end

function KLDCriterion:_viewInput(p, q)
   self.len = q:dim()
   if self.len == 3 then
      assert(p:size(2) == 2, "Incorrect dimension")
      self.nMix = q:size(2)/2
      self.qMu = q[{{},{1,self.nMix},{}}]
      self.qLv = q[{{},{self.nMix+1,2*self.nMix},{}}]
      if self.nMix > 1 then
         self.pMu = p[{{},{1},{}}]
         self.pLv = p[{{},{2},{}}]
      else
         self.pMu = p[{{},{1},{}}]
         self.pLv = p[{{},{2},{}}]
      end
   elseif self.len == 2 then
      assert(p:size(1) == 2, "Incorrect dimension")
      self.nMix = q:size(1)/2
      self.qMu = q[{{1,self.nMix},{}}]
      self.qLv = q[{{self.nMix+1,2*self.nMix},{}}]
      if self.nMix > 1 then
         self.pMu = p[{{1},{}}]
         self.pLv = p[{{2},{}}]
      else
         self.pMu = p[{{1},{}}]
         self.pLv = p[{{2},{}}]
      end
   end
end   
