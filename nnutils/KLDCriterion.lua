require 'nn'

local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

function KLDCriterion:__init()
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
end 

function KLDCriterion:updateOutput(p, q)
   -- p is batchSize x 2 x nDims
   -- q is batchSize x (2*nMixtures) x nDims
   local nMix, pMu, pLv, qMu, qLv
   local len = q:dim()
   if len == 3 then
      assert(p:size(2) == 2, "Incorrect dimension")
      nMix = q:size(2)/2
      qMu = q[{{},{1,nMix},{}}]
      qLv = q[{{},{nMix+1,2*nMix},{}}]
      if nMix > 1 then
         pMu = p[{{},1,{}}]:repeatTensor(1,nMix,1)
         pLv = p[{{},2,{}}]:repeatTensor(1,nMix,1)
      else
         pMu = p[{{},1,{}}]
         pLv = p[{{},2,{}}]
      end
   elseif len == 2 then
      assert(p:size(1) == 2, "Incorrect dimension")
      nMix = q:size(1)/2
      qMu = q[{{1,nMix},{}}]
      qLv = q[{{nMix+1,2*nMix},{}}]
      if nMix > 1 then
         pMu = p[{1,{}}]:repeatTensor(nMix,1)
         pLv = p[{2,{}}]:repeatTensor(nMix,1)
      else
         pMu = p[{1,{}}]
         pLv = p[{2,{}}]
      end
   end
   -- compute
   local lvDiff = qLv - pLv
   local muDiff = qMu - pMu
   local expDiff = (-lvDiff):exp()
   local qExp = (-qLv):exp()
   local expElem = torch.pow(muDiff, 2):cmul(qExp):add(expDiff):csub(1):add(lvDiff):div(2):neg():exp()
   local expSum = expElem:sum(len-1)
   self.output = (expSum/nMix):log():neg():sum()
   -- set self
   self.nMix = nMix
   self.len = len
   self.pMu = pMu
   self.pLv = pLv
   self.qMu = qMu
   self.qLv = qLv
   self.lvDiff = lvDiff
   self.muDiff = muDiff
   self.expDiff = expDiff
   self.qExp = qExp
   self.expElem = expElem
   self.expSum = expSum
   return self.output
end

function KLDCriterion:updateGradInput(p, q)
   local gradInput = self.gradInput
   local pMu = self.pMu
   local pLv = self.pLv
   local qMu = self.qMu
   local qLv = self.qLv
   local nMix = self.nMix
   local len = self.len
   local lvDiff = self.lvDiff
   local muDiff = self.muDiff
   local expDiff = self.expDiff
   local qExp = self.qExp
   local dElem = self.expElem
   local expSum = self.expSum
   gradInput[1]:resizeAs(q)
   gradInput[2]:resizeAs(q)
   local dpMu, dpLv, dqMu, dqLv
   if len == 3 then
      expSum = expSum:repeatTensor(1,nMix,1)
      dpMu = gradInput[1][{{},{1,nMix},{}}]
      dpLv = gradInput[1][{{},{nMix+1,2*nMix},{}}]
      dqMu = gradInput[2][{{},{1,nMix},{}}]
      dqLv = gradInput[2][{{},{nMix+1,2*nMix},{}}]
   elseif len == 2 then
      expSum = expSum:repeatTensor(nMix,1)
      dpMu = gradInput[1][{{1,nMix},{}}]
      dpLv = gradInput[1][{{nMix+1,2*nMix},{}}]
      dqMu = gradInput[2][{{1,nMix},{}}]
      dqLv = gradInput[2][{{nMix+1,2*nMix},{}}]
   end
   dElem:cdiv(expSum)
   dpMu:copy(muDiff):cmul(qExp):neg():cmul(dElem)
   dpLv:copy(expDiff):csub(1):div(2):cmul(dElem)
   dqMu:copy(muDiff):cmul(qExp):cmul(dElem)
   dqLv:copy(muDiff):pow(2):cmul(qExp):neg():csub(expDiff):add(1):div(2):cmul(dElem)
   if nMix > 1 then
      gradInput[1] = p.new():resizeAs(p)
      if len == 3 then
         gradInput[1][{{},1,{}}] = dpMu:sum(2)
         gradInput[1][{{},2,{}}] = dpLv:sum(2)
      elseif len == 2 then
         gradInput[1][{1,{}}] = dpMu:sum(1)
         gradInput[1][{2,{}}] = dpLv:sum(1)
      end
   end
   return self.gradInput[1], self.gradInput[2]
end
