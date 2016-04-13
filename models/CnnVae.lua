require 'models.Vae'
local CnnVae, parent = torch.class('CnnVae', 'Vae')
local c = require 'trepl.colorize'
require 'nngraph'
require 'nnutils.init'

function CnnVae:__init(struct)
   -- build model
   self.encoder, self.decoder, self.model = self:build(struct)
   self.kld = nn.KLDCriterion()
   self.bce = nn.BCECriterion()
   self.kldWeight = 1
   self.bceWeight = 1
   self.bce.sizeAverage = false
   self.parameters, self.gradients = self.model:getParameters()
end

function CnnVae:feval(x, minibatch)
   local input = minibatch[1]
   if self.parameters ~= x then
      self.parameters:copy(x)
   end
   self.model:zeroGradParameters()
   -- forward
   local mulv, recon = unpack(self.model:forward(input))
   local kld_err = self.kld:forward(mulv)
   local bce_err = self.bce:forward(recon, input)
   -- backward
   local dmulv = self.kld:backward(mulv, pmulv)
   local drecon = self.bce:backward(recon, input)
   error_grads = {dmulv:mul(self.kldWeight), drecon:mul(self.bceWeight)}
   self.model:backward(input, error_grads)
   -- record
   local nelbo = kld_err + bce_err
   self:record(kld_err, bce_err, nelbo)
   return nelbo, self.gradients
end

function CnnVae:cuda()
   parent.cuda(self)
   require 'cudnn'
   cudnn.benchmark = true
   cudnn.fastest = true
   cudnn.convert(self.encoder, cudnn)
   cudnn.convert(self.decoder, cudnn)
   return self
end
